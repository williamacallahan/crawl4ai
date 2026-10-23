import copy
import subprocess

import pytest

import rollout_policy
import rollout_tasks
from conftest import BASELINE, CANDIDATE, application, service_spec


def test_container_healthcheck_uses_liveness_and_public_proof_uses_readiness():
    assert rollout_policy.HEALTHCHECK["Test"] == [
        "CMD", "curl", "-f", "http://localhost:11235/health/live"
    ]
    assert all(url.endswith("/health") for url in rollout_policy.HEALTH_URLS)


def redis_tasks(state="Running 19 hours ago", node=None):
    return [
        {
            "ID": "redis1",
            "Name": "crawl4ai-redis.1",
            "Node": node or rollout_policy.REDIS_NODE,
            "DesiredState": "Running",
            "CurrentState": state,
        },
        {
            "ID": "redis0",
            "Name": "crawl4ai-redis.1",
            "Node": rollout_policy.REDIS_NODE,
            "DesiredState": "Shutdown",
            "CurrentState": "Shutdown 30 hours ago",
        },
    ]


def redis_spec():
    return {
        "TaskTemplate": {
            "ContainerSpec": {
                "Image": "redis@sha256:redis",
                "Command": list(rollout_policy.REDIS_COMMAND),
                "Mounts": copy.deepcopy(rollout_policy.REDIS_MOUNTS),
                "Healthcheck": {
                    "Test": list(rollout_policy.REDIS_HEALTHCHECK_TEST),
                    **copy.deepcopy(rollout_policy.REDIS_HEALTHCHECK_FLOORS),
                },
            },
            "Placement": {
                "Constraints": [f"node.hostname=={rollout_policy.REDIS_NODE}"],
                "MaxReplicas": 1,
            },
        },
        "Mode": {"Replicated": {"Replicas": 1}},
    }


def test_redis_pins_encode_the_2026_08_28_incident_invariants():
    """The drift check is only as good as what it pins, so pin the two values
    the incident was about rather than trusting the comparison alone."""
    assert "everysec" in rollout_policy.REDIS_COMMAND and "always" not in rollout_policy.REDIS_COMMAND
    assert rollout_policy.REDIS_COMMAND[rollout_policy.REDIS_COMMAND.index("--appendfsync") + 1] == "everysec"
    assert rollout_policy.REDIS_COMMAND[rollout_policy.REDIS_COMMAND.index("--appendonly") + 1] == "yes"
    assert rollout_policy.REDIS_COMMAND[rollout_policy.REDIS_COMMAND.index("--maxmemory") + 1] == "700mb"
    assert rollout_policy.REDIS_COMMAND[rollout_policy.REDIS_COMMAND.index("--maxmemory-policy") + 1] == "volatile-lru"
    assert rollout_policy.REDIS_MOUNTS == [
        {"Type": "volume", "Source": "crawl4ai-redis-data", "Target": "/data"}
    ]
    assert rollout_policy.REDIS_HEALTHCHECK_FLOORS["Retries"] >= 3
    assert rollout_policy.REDIS_HEALTHCHECK_FLOORS["Interval"] >= 5_000_000_000


def test_redis_proof_rejects_durability_drift(monkeypatch):
    def check(spec=None, tasks=None):
        monkeypatch.setattr(rollout_tasks, "_service_spec", lambda _n: copy.deepcopy(spec or redis_spec()))
        monkeypatch.setattr(rollout_tasks, "_service_tasks", lambda _n: copy.deepcopy(tasks or redis_tasks()))
        rollout_tasks._verify_redis()

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
    assert rollout_policy._drain_budget_ns(shipped["env"]) < rollout_policy.STOP_GRACE_NS
    rollout_policy._policy(shipped)

    # A drain delay raised in the Dokploy console must be what is measured,
    # not the fallback literal in entrypoint.sh.
    raised = application()
    raised["env"] += "\nCRAWL4AI_DRAIN_DELAY_SECONDS=600"
    assert rollout_policy._drain_budget_ns(raised["env"]) > rollout_policy.STOP_GRACE_NS
    with pytest.raises(ValueError, match="drain budget"):
        rollout_policy._policy(raised)

    monkeypatch.setattr(rollout_policy, "_drain_budget_ns", lambda _env: rollout_policy.STOP_GRACE_NS)
    with pytest.raises(ValueError, match="drain budget"):
        rollout_policy._policy(application())


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        # Pure-digit values str.isdigit() already accepted: no regression.
        ("0", 0), ("00", 0), ("2", 2), ("30", 30), ("40", 40), ("600", 600),
        ("007", 7),
        # Leading '+' the container's sleep honors but isdigit rejected.
        ("+2", 2), ("+0", 0),
        # Fractional numbers the container's sleep honors but isdigit rejected.
        ("1.5", 1.5), ("31.5", 31.5), (".5", 0.5), ("1.", 1), ("0.0", 0), ("10.0", 10),
        # Suffixed values: the realistic operator raise, all rejected by isdigit.
        ("30s", 30), ("2s", 2), ("1m", 60), ("2m", 120), ("1h", 3600), ("1d", 86400),
        ("0m", 0), ("0s", 0), ("0d", 0),
        # Fractional suffixed.
        ("1.5m", 90), (".5m", 30), ("1.m", 60), ("1.0m", 60), ("2.5h", 9000),
        ("1.5h", 5400), (".5s", 0.5),
        ("1.5e2", 150), ("0x10", 16),
        # Leading '+' with a suffix.
        ("+1m", 60), ("+1.5m", 90),
        # Leading whitespace the container's sleep honors (strtod skips it).
        # _drain_budget_ns used to .strip() the deployed value before parsing,
        # masking this shape from the strict regex; the leading \s* in
        # _SLEEP_INTERVAL mirrors sleep so the verifier never fails the deploy
        # on a value the container would honor.
        (" 2", 2), ("  2", 2), (" 5s", 5), (" 1m", 60), (" +2", 2), (" +1.5m", 90),
    ],
)
def test_parse_sleep_seconds_accepts_the_sleep_intervals_the_container_honors(value, seconds):
    # entrypoint.sh runs `sleep "${CRAWL4AI_DRAIN_DELAY_SECONDS:-2}"`, so the
    # verifier must measure any single argument the container's sleep accepts;
    # isdigit rejected every non-digit row below, masking it with the 2-literal.
    assert rollout_policy._parse_sleep_seconds(value) == seconds


@pytest.mark.parametrize(
    "value",
    [
        # Not a number.
        "abc", "m", "s", "",
        # Units the container's sleep rejects.
        "1y", "1w", "1M", "1S", "1x",
        # Multi-component single arguments (sleep wants separate args).
        "1m30s", "1d2h", "1 m",
        # Malformed decimals.
        "1.5.5", "1.5.", "1..5", "1.5.5m", "+.", "+", "1_000",
        # Negative: sleep treats as an option, and a negative drain is meaningless.
        "-2", "-2s",
        # Exotic forms strtod accepts but a drain delay must never be: a huge or
        # infinite sleep would always SIGKILL the task, so reject loudly so the
        # operator's typo can never slip past the gate unmeasured.
        "inf", "nan",
        # Trailing or wrapped whitespace: GNU sleep rejects these with exit 1
        # (trailing whitespace is the one shape strtod does NOT skip), and so
        # must the verifier. _drain_budget_ns used to .strip() these into clean
        # numbers that passed the gate — the exact data-loss class the
        # STOP_GRACE_NS gate exists to prevent. The leading \s* in
        # _SLEEP_INTERVAL must not bleed into trailing tolerance: fullmatch with
        # no trailing \s* keeps these rejected.
        "2 ", "  2  ", " 5 ", "5s ", " 5s ", "2\t",
    ],
)
def test_parse_sleep_seconds_rejects_non_intervals_loudly(value):
    with pytest.raises(ValueError, match="CRAWL4AI_DRAIN_DELAY_SECONDS"):
        rollout_policy._parse_sleep_seconds(value)


@pytest.mark.parametrize("drain", [None, "", "2"])
def test_drain_budget_ns_falls_back_only_when_the_delay_is_unset(drain):
    # 360 (stopwaitsecs) + 2 (entrypoint literal) == 362s: an unset variable and
    # a value the grace easily covers both keep the shipped default budget.
    env = "IGNORE=me\n"
    if drain is not None:
        env += f"CRAWL4AI_DRAIN_DELAY_SECONDS={drain}"
    assert rollout_policy._drain_budget_ns(env) == 362_000_000_000


@pytest.mark.parametrize(
    "drain",
    [
        "abc", "1m30s", "1d2h", "-2", "1.5.5", "1M", "1_000", "inf", "nan",
        # Whitespace-padded numeric values the container's sleep rejects but
        # _drain_budget_ns used to .strip() into clean numbers that passed the
        # gate (the bug): trailing-or-wrapped whitespace aborts `sleep` with
        # exit 1 at drain time, so the trap SIGKILLs before supervisord drains.
        "2 ", "  2  ", " 5 ", "5s ", " 5s ", "2\t",
    ],
)
def test_drain_budget_ns_raises_loudly_when_a_set_delay_is_unparseable(drain):
    # The original bug fell back to the entrypoint literal 2 whenever isdigit
    # rejected a set value; a value that is set but unparseable must fail the
    # deploy loudly instead of silently passing a gate on a budget it never
    # measured.
    with pytest.raises(ValueError, match="CRAWL4AI_DRAIN_DELAY_SECONDS"):
        rollout_policy._drain_budget_ns(f"CRAWL4AI_DRAIN_DELAY_SECONDS={drain}")


@pytest.mark.parametrize(
    "drain", ["1m", "30s", "31.5", "1.5m", "2m", "1h", "40", "600"]
)
def test_policy_blocks_a_rollout_whose_drain_delay_meets_or_exceeds_the_stop_grace(drain):
    raised = application()
    raised["env"] += f"\nCRAWL4AI_DRAIN_DELAY_SECONDS={drain}"
    with pytest.raises(ValueError, match="drain budget"):
        rollout_policy._policy(raised)


@pytest.mark.parametrize(
    "drain", ["2 ", "  2  ", " 5 ", "5s ", " 5s ", "2\t"]
)
def test_drain_budget_ns_rejects_trailing_whitespace_the_containers_sleep_rejects(drain):
    # Regression for the .strip() bug: _drain_budget_ns used to .strip() the
    # deployed value before _parse_sleep_seconds saw it, so "2 " or "  2  "
    # was re-formed to "2" and passed the gate. The container's
    # `sleep "2 "` exits 1 (GNU sleep rejects trailing whitespace — the one
    # shape strtod does not skip), which under `set -e` in entrypoint.sh's
    # begin_drain trap aborts before `kill -TERM supervisord`, so the task is
    # SIGKILLed mid-drain — the data-loss class the STOP_GRACE_NS gate exists
    # to prevent. _drain_budget_ns must fail the deploy loudly on the raw
    # value sleep will actually receive.
    sleep_rejects = subprocess.run(
        ["sleep", drain], check=False
    ).returncode != 0
    assert sleep_rejects, f"container sleep unexpectedly accepted {drain!r}"
    with pytest.raises(ValueError, match="CRAWL4AI_DRAIN_DELAY_SECONDS"):
        rollout_policy._drain_budget_ns(f"CRAWL4AI_DRAIN_DELAY_SECONDS={drain}")


@pytest.mark.parametrize(
    ("drain", "seconds"),
    [(" 0", 0), (" 0s", 0), (" 2", 2), (" +2", 2), (" 5s", 5), (" 1m", 60)]
)
def test_drain_budget_ns_honors_leading_whitespace_the_containers_sleep_honors(
    drain, seconds,
):
    # Parity guard: GNU sleep skips leading whitespace (a strtod artifact), so
    # " 2" sleeps 2 seconds. The pre-bug _drain_budget_ns .strip()ed the value
    # to "2" and passed the gate that way; the fix keeps the raw value and
    # relies on _SLEEP_INTERVAL's leading \s* to model sleep faithfully, so a
    # deploy is never failed loudly on a value the container would honor.
    # Cross-check the container's sleep end-to-end only for 0-second rows (a
    # nonzero interval would stall the suite); strtod's leading-whitespace
    # handling is uniform, so the 0-second proof generalizes to " 2", " 5s"
    # and " 1m", whose verifier side is pinned by the budget assertion below.
    if seconds == 0:
        sleep_exit = subprocess.run(["sleep", drain], check=False).returncode
        assert sleep_exit == 0, f"container sleep rejected {drain!r} (exit {sleep_exit})"
    budget = rollout_policy._drain_budget_ns(
        f"LLM_PROVIDER={rollout_policy.LLM_PROVIDER}\n"
        f"LLM_BASE_URL={rollout_policy.LLM_BASE_URL}\n"
        f"LLM_API_KEY=secret\nCRAWL4AI_DRAIN_DELAY_SECONDS={drain}"
    )
    assert budget == (360 + seconds) * 1_000_000_000


def test_drain_budget_ns_passes_the_raw_value_untrimmed_to_the_parser(monkeypatch):
    # The bug was a .strip() between _environment_values and _parse_sleep_seconds
    # that re-formed "2 " into "2". Lock the contract: the parser receives the
    # exact bytes the container's sleep will receive, so any future re-introduction
    # of an upstream trim is caught immediately.
    seen = {}
    def fake_parse(value):
        seen["value"] = value
        raise ValueError("stop")
    monkeypatch.setattr(rollout_policy, "_parse_sleep_seconds", fake_parse)
    with pytest.raises(ValueError, match="stop"):
        rollout_policy._drain_budget_ns(
            "LLM_PROVIDER=x\nLLM_BASE_URL=y\nLLM_API_KEY=k"
            "\nCRAWL4AI_DRAIN_DELAY_SECONDS=  2  "
        )
    assert seen["value"] == "  2  "


def test_policy_accepts_only_stock_docker_image_configuration():
    rollout_policy._policy(application())
    for field, value in (
        ("sourceType", "github"),
        ("replicas", 2),
        ("healthCheckSwarm", {}),
        ("placementSwarm", {}),
        ("endpointSpecSwarm", {}),
        ("updateConfigSwarm", {}),
        ("rollbackConfigSwarm", {}),
        ("stopGracePeriodSwarm", 1),
        ("stopGracePeriodSwarm", rollout_policy.STOP_GRACE_NS + 1),
        ("cpuReservation", "1"),
        ("cpuLimit", "1"),
        ("memoryReservation", "1"),
        ("memoryLimit", "1"),
    ):
        changed = application()
        changed[field] = value
        with pytest.raises(ValueError):
            rollout_policy._policy(changed)

    changed = application()
    changed["env"] = changed["env"].replace(
        rollout_policy.LLM_BASE_URL, "https://attacker.invalid/v1"
    )
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        rollout_policy._policy(changed)

    changed = application()
    changed["env"] += f"\nLLM_BASE_URL={rollout_policy.LLM_BASE_URL}"
    with pytest.raises(ValueError, match="duplicate LLM_BASE_URL"):
        rollout_policy._policy(changed)


def test_running_spec_rejects_artifact_drift():
    rollout_policy._running_spec(service_spec(), BASELINE, rollout_policy._labels("baseline"))
    changed = service_spec()
    changed["TaskTemplate"]["ContainerSpec"]["Image"] = CANDIDATE
    with pytest.raises(ValueError, match="artifact"):
        rollout_policy._running_spec(changed, BASELINE, rollout_policy._labels("baseline"))

    changed = service_spec()
    changed["TaskTemplate"]["ContainerSpec"]["Env"][1] = (
        "LLM_BASE_URL=https://attacker.invalid/v1"
    )
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        rollout_policy._running_spec(changed, BASELINE, rollout_policy._labels("baseline"))

    changed = service_spec()
    changed["TaskTemplate"]["ContainerSpec"]["Env"].append(
        f"LLM_BASE_URL={rollout_policy.LLM_BASE_URL}"
    )
    with pytest.raises(ValueError, match="duplicate LLM_BASE_URL"):
        rollout_policy._running_spec(changed, BASELINE, rollout_policy._labels("baseline"))

    changed = service_spec()
    changed["TaskTemplate"]["ContainerSpec"]["StopGracePeriod"] = (
        rollout_policy.STOP_GRACE_NS + 1
    )
    with pytest.raises(ValueError, match="stop grace"):
        rollout_policy._running_spec(changed, BASELINE, rollout_policy._labels("baseline"))


def test_running_spec_requires_record_labels_beside_control_plane_labels():
    # Dokploy stamps dokploy.deployment.id on every rendered ContainerSpec since
    # its 2026-09-01 build; exact label equality then failed every rollout (runs
    # 33541789332 and 33544931205) although the record's own labels were intact.
    stamped = service_spec()
    stamped["TaskTemplate"]["ContainerSpec"]["Labels"]["dokploy.deployment.id"] = "row"
    rollout_policy._running_spec(stamped, BASELINE, rollout_policy._labels("baseline"))

    for missing_or_changed in ("otel.service.version", "otel.logs.enabled"):
        changed = service_spec()
        changed["TaskTemplate"]["ContainerSpec"]["Labels"]["dokploy.deployment.id"] = "row"
        changed["TaskTemplate"]["ContainerSpec"]["Labels"][missing_or_changed] = "other"
        with pytest.raises(ValueError, match="labels"):
            rollout_policy._running_spec(changed, BASELINE, rollout_policy._labels("baseline"))
        del changed["TaskTemplate"]["ContainerSpec"]["Labels"][missing_or_changed]
        with pytest.raises(ValueError, match="labels"):
            rollout_policy._running_spec(changed, BASELINE, rollout_policy._labels("baseline"))


def test_policy_accepts_capped_placement():
    app = application()
    app["placementSwarm"] = copy.deepcopy(rollout_policy._CAPPED_PLACEMENT)
    rollout_policy._policy(app)


@pytest.mark.parametrize("placement", [
    {"Constraints": ["node.role==manager"]},
    {**rollout_policy.PLACEMENT, "Preferences": [{"Spread": {"SpreadDescriptor": "node.role"}}]},
])
def test_policy_rejects_foreign_placement(placement):
    app = application(placement=placement)
    with pytest.raises(ValueError, match="placement drifted"):
        rollout_policy._policy(app)


def test_running_spec_accepts_capped_placement():
    spec = service_spec()
    spec["TaskTemplate"]["Placement"] = copy.deepcopy(rollout_policy._CAPPED_PLACEMENT)
    rollout_policy._running_spec(spec, BASELINE, rollout_policy._labels("baseline"))


def test_running_spec_requires_the_exact_record_placement():
    # An accepted placement still must match the provider record.
    spec = service_spec()
    spec["TaskTemplate"]["Placement"] = copy.deepcopy(rollout_policy._CAPPED_PLACEMENT)
    with pytest.raises(ValueError, match="placement drifted"):
        rollout_policy._running_spec(
            spec, BASELINE, rollout_policy._labels("baseline"), placements=(rollout_policy.PLACEMENT,)
        )
