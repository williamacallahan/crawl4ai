"""Read and drive the stock Dokploy application record and its deployments."""

from __future__ import annotations

import subprocess
import time
from typing import Any

import rollout_withdrawals as withdrawals
import rollout_http
import rollout_policy
import rollout_tasks


def _heal_stranded_record(
    base: str,
    api_key: str,
    application_id: str,
    app_name: str,
    application: dict[str, Any],
) -> dict[str, Any]:
    """Realign a record left describing an image the service never ran.

    Dokploy deploys FROM the record, so a deploy must patch the record to the
    candidate before the service can move. When anything after that fails, the
    record is stranded on the candidate while the service still runs the
    baseline, and the next run dies in _running_spec with "running artifact or
    labels differ from Dokploy" - a different error than the real one, which is
    how one failure turns into a chain of unrelated ones (2026-09-04, run
    33852834121, which needed a manual REST repair).

    The failing run must NOT repair this itself: a failed write has an ambiguous
    outcome, and compensating for it is exactly what
    test_deploy_does_not_compensate_for_ambiguous_write forbids. Healing belongs
    here instead, at the start of the next run, where nothing is in flight - the
    Swarm update is terminal and no Dokploy deployment is running - so the live
    service is authoritative and the repair is unambiguous.
    """
    running = rollout_tasks._service_spec(app_name)["TaskTemplate"]["ContainerSpec"]
    running_image = str(running.get("Image", ""))
    if application.get("dockerImage") == running_image:
        return application
    running_revision = (running.get("Labels") or {}).get("otel.service.version")
    if "@sha256:" not in running_image or not isinstance(running_revision, str):
        raise ValueError("running service has no immutable artifact to heal from")
    if any(
        row.get("status") not in {"done", "error", "cancelled"}
        for row in _deployments(base, api_key, application_id)
    ):
        raise RuntimeError("Crawl4AI already has a nonterminal Dokploy deployment")
    print(
        f"record points at {application.get('dockerImage')} but the service runs "
        f"{running_image}; healing the record to the running artifact"
    )
    rollout_http._post_json(
        f"{base.rstrip('/')}/api/application.update",
        api_key,
        {
            "applicationId": application_id,
            "dockerImage": running_image,
            "labelsSwarm": rollout_policy._labels(running_revision),
        },
    )
    healed = _application(base, api_key, application_id)
    rollout_policy._policy(healed)
    if healed.get("dockerImage") != running_image or healed.get("labelsSwarm") != rollout_policy._labels(
        running_revision
    ):
        raise RuntimeError("record did not converge on the running artifact")
    return healed


def _ensure_rollback_source(image: str) -> None:
    """Make sure the deploy host holds the image Dokploy will tag for rollback.

    Dokploy's rollback-registry step runs `docker tag` against the *currently
    running* service image, on the Dokploy host, before it touches the service.
    That image is only ever there by accident: whichever runner built it keeps a
    copy, and `runs-on: [self-hosted, Linux, X64, williamacallahan]` can place
    the X64 build on any eligible runner. When haiku-0 does not have it the tag
    fails, the deployment ends `error`, and the Swarm service is never updated
    (2026-09-02 tag v63, 2026-09-04 tag v75).

    This job runs on canonical-haiku-0, which *is* the Dokploy host, so pulling
    here puts the image exactly where the tag will look for it. The digest moves
    with every rollout, so it cannot be seeded once and left alone.
    """
    present = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if present.returncode == 0:
        return
    pull = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
    if pull.returncode != 0:
        # Surface the reason: swallowing it turns an auth or manifest problem
        # into a bare CalledProcessError with no way to tell them apart.
        raise RuntimeError(
            f"could not seed the rollback source {image}: {pull.stderr.strip()}"
        )


def _application(base: str, api_key: str, application_id: str) -> dict[str, Any]:
    application = rollout_http._request_json(
        rollout_http._url(base, "application.one", applicationId=application_id), api_key
    )
    if not isinstance(application, dict):
        raise ValueError("invalid application response")
    return application


def _deployments(base: str, api_key: str, application_id: str) -> list[dict[str, Any]]:
    for attempt in range(3):
        try:
            deployments = rollout_http._request_json(
                rollout_http._url(base, "deployment.all", applicationId=application_id), api_key
            )
            break
        except rollout_http.CurlError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    if not isinstance(deployments, list):
        raise ValueError("invalid deployment response")
    return deployments


def _wait_deployment(
    base: str,
    api_key: str,
    application_id: str,
    prior_ids: set[str],
    title: str,
    description: str,
    collector: withdrawals.Withdrawals | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + rollout_policy.TIMEOUT_SECONDS
    if collector is not None:
        # Refresh at the same boundary the caller's own deadline does, so a
        # queue wait within its own TIMEOUT_SECONDS bound never fails a
        # collector whose runner deadline the earlier arming consumed.
        collector.refresh()
    admitted = False
    while time.monotonic() < deadline:
        try:
            rows = _deployments(base, api_key, application_id)
        except rollout_http.CurlError:
            # The control plane returns 502 under load, and outlasting the
            # short read retry used to abandon a deployment that was already
            # running. The deadline above still bounds this wait, and an
            # unreadable deployment stays unproven either way.
            time.sleep(5)
            continue
        new = [
            row
            for row in rows
            if row.get("deploymentId") not in prior_ids
        ]
        matching = [
            row
            for row in new
            if row.get("title") == title and row.get("description") == description
        ]
        if len(new) > 1 or len(matching) > 1 or (new and not matching):
            raise RuntimeError("deployment submission is ambiguous; no recovery write is allowed")
        if matching:
            owned = matching[0]
            if not admitted:
                admitted = True
                deadline = time.monotonic() + rollout_policy.TIMEOUT_SECONDS
                if collector is not None:
                    # The processing phase gets its own fresh caller bound; the
                    # runner keeps pace so a slow processing wait reports its
                    # own error, not a stale queue-admission collector expiry.
                    collector.refresh()
                print("stock Dokploy deployment was admitted; waiting for processing", flush=True)
            status = owned.get("status")
            if status == "done":
                return owned
            if status in {"error", "cancelled"}:
                raise withdrawals.StateError(f"stock Dokploy deployment ended in {status}", status)
        if collector is not None:
            # Surface a collector death inside the wait that owns it instead of
            # at the finish barrier, so a journal that expired or a collector
            # that failed during queue/processing is attributed to this phase.
            collector.check()
        time.sleep(5)
    if admitted:
        raise TimeoutError("stock Dokploy deployment did not finish after queue admission")
    raise TimeoutError("stock Dokploy deployment was not admitted from the queue")


def _record_converged(
    app: dict[str, Any], candidate: str, revision: str, placement: dict[str, Any]
) -> bool:
    return (
        app.get("dockerImage") == candidate
        and app.get("labelsSwarm") == rollout_policy._labels(revision)
        and app.get("placementSwarm") == placement
    )
