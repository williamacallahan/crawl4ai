"""Deploy one immutable Crawl4AI image through stock Dokploy and Swarm."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import rollout_withdrawals as withdrawals
import rollout_dokploy
import rollout_evidence
import rollout_http
import rollout_policy
import rollout_tasks


def _deploy() -> None:
    base = os.environ["DOKPLOY_URL"]
    api_key = os.environ["DOKPLOY_API_KEY"]
    application_id = os.environ["APPLICATION_ID"]
    revision = os.environ["GITHUB_SHA"]
    candidate = f"{os.environ['IMAGE']}@{os.environ['IMAGE_DIGEST']}"
    application = rollout_dokploy._application(base, api_key, application_id)
    app_name = str(application["appName"])
    # A native environment reload can outlive the CI build. Join its bounded
    # wait before capturing the baseline; never submit a competing update.
    deadline = time.monotonic() + rollout_policy.TIMEOUT_SECONDS
    state, _ = rollout_tasks._update_state(app_name)
    waited = False
    while state not in {"", "completed"}:
        if state not in {"updating", "rollback_started"}:
            if state in {"paused", "rollback_paused", "rollback_completed"}:
                print(f"prior Crawl4AI Swarm update ended in {state}; proving the running baseline", flush=True)
                break
            raise withdrawals.StateError(f"prior Crawl4AI Swarm update ended in {state}", state)
        if time.monotonic() >= deadline:
            raise TimeoutError("prior Crawl4AI Swarm update did not finish")
        if not waited:
            print("waiting for the prior native Crawl4AI Swarm update", flush=True)
            waited = True
        time.sleep(5)
        state, _ = rollout_tasks._update_state(app_name)
    if waited:
        application = rollout_dokploy._application(base, api_key, application_id)
    rollout_policy._policy(application)
    baseline = str(application.get("dockerImage", ""))
    baseline_labels = application.get("labelsSwarm")
    if "@sha256:" not in baseline or not isinstance(baseline_labels, dict):
        raise ValueError("stock Dokploy baseline is not immutable")
    baseline_revision = baseline_labels.get("otel.service.version")
    if not isinstance(baseline_revision, str) or not baseline_revision:
        raise ValueError("stock Dokploy baseline has no revision")
    # Nothing is in flight now, so a record that disagrees with the service is a
    # previous run's stranded candidate, not a live deploy. Repair it here.
    application = rollout_dokploy._heal_stranded_record(base, api_key, application_id, app_name, application)
    baseline = str(application.get("dockerImage", ""))
    baseline_labels = application.get("labelsSwarm")
    baseline_revision = (baseline_labels or {}).get("otel.service.version")
    if not isinstance(baseline_revision, str) or not baseline_revision:
        raise ValueError("stock Dokploy baseline has no revision")
    placement = application["placementSwarm"]
    running_spec = rollout_tasks._service_spec(app_name)
    rollout_policy._running_spec(running_spec, baseline, baseline_labels, placements=(placement,))
    rollout_dokploy._ensure_rollback_source(
        str(running_spec["TaskTemplate"]["ContainerSpec"]["Image"])
    )
    rollout_tasks._verify_redis()
    rollout_tasks.verify_route(base, api_key, application_id, app_name)
    # Captured adjacent to its use: this snapshot decides which unconfirmable
    # predecessors the baseline may excuse.
    eligibility = rollout_tasks._eligible_nodes()
    eligible, ready = eligibility
    already_deployed = rollout_dokploy._record_converged(application, candidate, revision, placement)
    census = (
        f"{rollout_policy.REPLICAS} replicas, {len(ready)} Ready ({', '.join(sorted(ready)) or 'none'}), "
        f"not Ready: {', '.join(sorted(eligible - ready)) or 'none'}"
    )
    if not already_deployed and len(ready) < rollout_policy.REPLICAS:
        raise RuntimeError(f"not enough Ready eligible nodes to place every replica: {census}")
    # Only a one-per-node cap makes a spare mandatory for start-first.
    if not already_deployed and placement.get("MaxReplicas") == 1 and len(ready) <= rollout_policy.REPLICAS:
        raise RuntimeError(f"start-first needs a spare eligible node: {census}")
    baseline_proof = rollout_tasks._verify_tasks(
        app_name, baseline, baseline_revision, eligibility, False, placement
    )
    for url in rollout_policy.HEALTH_URLS:
        if not rollout_tasks._exact_health(rollout_http._request_json(f"{url}?baseline={uuid.uuid4()}"), baseline_revision):
            raise RuntimeError("public Crawl4AI baseline is not ready")
    prior_deployments = rollout_dokploy._deployments(base, api_key, application_id)
    if any(
        row.get("status") not in {"done", "error", "cancelled"}
        for row in prior_deployments
    ):
        raise RuntimeError("Crawl4AI already has a nonterminal Dokploy deployment")
    # A rerun of an already-deployed revision has nothing to submit: Dokploy
    # makes no spec change for an image the record already carries, so it opens
    # no deployment row and the wait below would time out on a row that never
    # appears. The baseline proof above already ran against this same candidate,
    # so the proof below is the whole remaining job.
    deployment_id = None
    withdrawal_proof = None
    collector = None
    original_census = None
    if already_deployed:
        print("candidate is already deployed; proving it in place", flush=True)
    else:
        prior_ids = {str(row.get("deploymentId")) for row in prior_deployments}
        title = f"crawl4ai-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}-{revision}"
        description = f"candidate={candidate};baseline={baseline}"
        current = rollout_dokploy._application(base, api_key, application_id)
        rollout_policy._policy(current)
        if (
            current.get("dockerImage") != baseline
            or current.get("labelsSwarm") != baseline_labels
            or current.get("placementSwarm") != placement
        ):
            raise RuntimeError("baseline metadata changed before submission")
        rollout_tasks.verify_route(base, api_key, application_id, app_name)
        original_census = withdrawals.census(app_name)
        collector = withdrawals.Withdrawals(
            app_name, baseline, Path(f"{rollout_tasks._task_proof_path()}.withdrawals.json"),
            rollout_policy.TIMEOUT_SECONDS, rollout_policy.WITHDRAWAL_COLLECT_SECONDS
        )
        if collector.start() != original_census:
            raise RuntimeError("baseline task identity changed while arming withdrawal proof")
        rollout_policy._running_spec(rollout_tasks._service_spec(app_name), baseline, baseline_labels, placements=(placement,))
        if rollout_tasks._verify_tasks(app_name, baseline, baseline_revision, eligibility, False, placement) != baseline_proof:
            raise RuntimeError("baseline readiness changed while arming withdrawal proof")
        if withdrawals.ready_nodes() != collector.nodes:
            raise RuntimeError("eligible node coverage changed before candidate submission")
        current = rollout_dokploy._application(base, api_key, application_id)
        rollout_policy._policy(current)
        if (current.get("dockerImage") != baseline
                or current.get("labelsSwarm") != baseline_labels
                or current.get("placementSwarm") != placement):
            raise RuntimeError("baseline metadata changed while arming withdrawal proof")
        rollout_tasks.verify_route(base, api_key, application_id, app_name)
        _, prior_started_at = rollout_tasks._update_state(app_name)
        rollout_http._post_json(
            f"{base.rstrip('/')}/api/application.update",
            api_key,
            {
                "applicationId": application_id,
                "dockerImage": candidate,
                "labelsSwarm": rollout_policy._labels(revision),
                "placementSwarm": placement,
            },
        )
        updated = rollout_dokploy._application(base, api_key, application_id)
        rollout_policy._policy(updated)
        if not rollout_dokploy._record_converged(updated, candidate, revision, placement):
            raise RuntimeError("candidate metadata did not converge; no deploy was submitted")
        rollout_tasks.verify_route(base, api_key, application_id, app_name)
        rollout_http._post_json(
            f"{base.rstrip('/')}/api/application.deploy",
            api_key,
            {"applicationId": application_id, "title": title, "description": description},
        )
        deployment_id = rollout_dokploy._wait_deployment(
            base, api_key, application_id, prior_ids, title, description, collector
        )["deploymentId"]
        deadline = time.monotonic() + rollout_policy.TIMEOUT_SECONDS
        # The Swarm-update wait has its own fresh caller bound; refresh the
        # runner's deadline to match so a slow update reports its own error
        # rather than failing a collector whose runner deadline the queue wait
        # consumed.
        collector.refresh()
        while True:
            state, started_at = rollout_tasks._update_state(app_name)
            # Swarm can still expose the previous terminal result while
            # Dokploy's newly submitted update is waiting to start.
            current_update = started_at != "" and started_at != prior_started_at
            if current_update and state == "completed":
                break
            if current_update and state in {"paused", "rollback_paused", "rollback_completed"}:
                raise withdrawals.StateError(
                    f"Swarm update ended in {state}; manual reconciliation required", state
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("Swarm update did not reach completed")
            # Fail fast on a collector death (or its internal journal expiry)
            # inside the wait that owns the slow update rather than at the
            # finish barrier after a successful rollout.
            collector.check()
            time.sleep(5)
        # The public proofs and the finish barrier each take their own fresh
        # caller span; refresh so a slow proof phase does not fail the
        # collector for the time the Swarm-update wait consumed.
        collector.refresh()
    final = rollout_dokploy._application(base, api_key, application_id)
    rollout_policy._policy(final)
    if not rollout_dokploy._record_converged(final, candidate, revision, placement):
        raise RuntimeError("foreign application metadata replaced the candidate")
    rollout_policy._running_spec(
        rollout_tasks._service_spec(app_name), candidate, rollout_policy._labels(revision), placements=(placement,)
    )
    eligibility = rollout_tasks._eligible_nodes()
    _, ready = eligibility
    if len(ready) < rollout_policy.REPLICAS:
        raise RuntimeError("not enough Ready eligible nodes to place every replica")
    # Real rollouts require clean predecessor withdrawals. A no-op rerun
    # proves the live revision without attributing older withdrawals to this run.
    converged = not already_deployed
    task_proof = rollout_tasks._verify_tasks(
        app_name, candidate, revision, eligibility, converged, placement
    )
    rollout_tasks.verify_route(base, api_key, application_id, app_name)
    public_instances = rollout_tasks._verify_public(revision, task_proof["instances"])
    post_public = rollout_dokploy._application(base, api_key, application_id)
    rollout_policy._policy(post_public)
    if not rollout_dokploy._record_converged(post_public, candidate, revision, placement):
        raise RuntimeError("candidate metadata changed during public proof")
    rollout_policy._running_spec(
        rollout_tasks._service_spec(app_name), candidate, rollout_policy._labels(revision), placements=(placement,)
    )
    rollout_tasks.verify_route(base, api_key, application_id, app_name)
    if rollout_tasks._verify_tasks(
        app_name, candidate, revision, eligibility, converged, placement
    ) != task_proof:
        raise RuntimeError("Crawl4AI task census changed during public proof")
    if collector is not None:
        # The finish barrier gets its own fresh caller span: each collector's
        # end marker and "complete" log appear within seconds, so a refresh
        # here keeps the runner's deadline from misreporting a slow earlier
        # phase as the final collector `"expired"` error.
        collector.refresh()
        withdrawal_proof = collector.finish(withdrawals.census(app_name))
    proof = {
        "revision": revision,
        "image": candidate,
        "baselineRevision": baseline_revision,
        "placement": placement,
        **task_proof,
        "publicInstances": public_instances,
        "withdrawals": withdrawal_proof,
        "baselineTasks": original_census,
        "rolloutSubmitted": not already_deployed,
    }
    rollout_tasks._task_proof_path().write_text(json.dumps(proof, separators=(",", ":")))
    print(
        json.dumps(
            {
                "deploymentId": deployment_id,
                **proof,
            },
            separators=(",", ":"),
        )
    )


def deploy() -> None:
    try:
        _deploy()
    finally:
        if "ROLLOUT_MONITOR_PATH" in os.environ:
            cleanup_withdrawals()


def cleanup_withdrawals() -> None:
    withdrawals.cleanup(Path(f"{rollout_tasks._task_proof_path()}.withdrawals.json"))


def _public_failure(command: str, error: Exception) -> str:
    """Name the failure class for the public CI log without estate data.

    The verifier check is the innermost frame of this package. The message is
    shown only when it appears verbatim as a source literal: an interpolated
    message can carry node names, image digests, or curl's host diagnosis.
    """
    frames = traceback.extract_tb(error.__traceback__)
    own = [frame for frame in frames if Path(frame.filename).parent == Path(__file__).parent]
    fields = [
        f"command={command}",
        f"check={own[-1].name if own else 'unknown'}",
        f"error={type(error).__name__}",
    ]
    if isinstance(error, withdrawals.StateError):
        fields.append(f"state={error.state}")
    exit_code = (
        error.curl_exit if isinstance(error, rollout_http.CurlError)
        else error.returncode if isinstance(error, subprocess.CalledProcessError)
        else error.exit_code if isinstance(error, withdrawals.StateError)
        else None
    )
    if exit_code is not None:
        fields.append(f"exit={exit_code}")
    message = json.dumps(str(error))
    raised_here = bool(own) and own[-1] is frames[-1]
    if raised_here and message in Path(frames[-1].filename).read_text():
        fields.append(f"reason={message}")
    return "rollout failure: " + " ".join(fields)


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "deploy"
    try:
        {"deploy": deploy, "monitor": rollout_evidence.monitor, "evidence": rollout_evidence.evidence,
         "cleanup-withdrawals": cleanup_withdrawals}[command]()
    except Exception as error:
        try:
            line = _public_failure(command, error)
        except Exception:
            line = f"rollout failure: command={command} error={type(error).__name__}"
        print(line, file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
