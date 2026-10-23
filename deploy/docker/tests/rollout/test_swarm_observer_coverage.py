import subprocess
from dataclasses import fields

import pytest

import rollout_policy
import swarm_observer as observer


OBSERVER_MISMATCH = observer.NetworkDbFdbComparison.DESTINATION_MISMATCH


OBSERVER_COMPLETE = observer.CoverageSnapshot(
    rollout_policy.REPLICAS, rollout_policy.REPLICAS, rollout_policy.REPLICAS
)


OBSERVER_INCOMPLETE = observer.CoverageSnapshot(3, 2, 1)


def test_initial_incomplete_is_reported_once_until_full_recovery(tmp_path):
    latch = observer.CoverageEpisodeLatch(tmp_path / "coverage-state")

    assert latch.observe(
        OBSERVER_INCOMPLETE, OBSERVER_MISMATCH
    ) == observer.CoverageDiagnostic(3, 2, 1, OBSERVER_MISMATCH)
    latch.acknowledge()
    assert (
        latch.observe(OBSERVER_INCOMPLETE, observer.NetworkDbFdbComparison.MISSING)
        is None
    )
    assert (
        latch.observe(OBSERVER_COMPLETE, observer.NetworkDbFdbComparison.CONSISTENT)
        is None
    )
    assert latch.observe(
        OBSERVER_INCOMPLETE, observer.NetworkDbFdbComparison.MISSING
    ) == observer.CoverageDiagnostic(3, 2, 1, observer.NetworkDbFdbComparison.MISSING)


def test_full_to_incomplete_is_reported_once_until_recovery(tmp_path):
    latch = observer.CoverageEpisodeLatch(tmp_path / "coverage-state")

    assert (
        latch.observe(OBSERVER_COMPLETE, observer.NetworkDbFdbComparison.CONSISTENT)
        is None
    )
    assert latch.observe(
        OBSERVER_INCOMPLETE, OBSERVER_MISMATCH
    ) == observer.CoverageDiagnostic(3, 2, 1, OBSERVER_MISMATCH)
    latch.acknowledge()
    assert (
        latch.observe(OBSERVER_INCOMPLETE, observer.NetworkDbFdbComparison.INCONCLUSIVE)
        is None
    )
    assert (
        latch.observe(OBSERVER_COMPLETE, observer.NetworkDbFdbComparison.CONSISTENT)
        is None
    )


def test_latch_state_survives_a_restart_and_resets_after_recovery(tmp_path):
    state_path = tmp_path / "coverage-state"
    first = observer.CoverageEpisodeLatch(state_path)

    assert first.observe(OBSERVER_INCOMPLETE, OBSERVER_MISMATCH) is not None
    assert state_path.read_bytes() == b"pending\n"
    restarted = observer.CoverageEpisodeLatch(state_path)
    assert restarted.observe(OBSERVER_INCOMPLETE, OBSERVER_MISMATCH) is not None
    restarted.acknowledge()
    assert state_path.read_bytes() == b"open\n"
    emitted = observer.CoverageEpisodeLatch(state_path)
    assert emitted.observe(OBSERVER_INCOMPLETE, OBSERVER_MISMATCH) is None
    assert (
        emitted.observe(OBSERVER_COMPLETE, observer.NetworkDbFdbComparison.CONSISTENT)
        is None
    )
    assert state_path.read_bytes() == b"closed\n"
    recovered = observer.CoverageEpisodeLatch(state_path)
    assert recovered.observe(OBSERVER_INCOMPLETE, OBSERVER_MISMATCH) is not None


def test_coverage_metrics_are_fixed_unlabeled_and_zero_is_incomplete():
    snapshot = observer.CoverageSnapshot(3, 2, 1)

    assert snapshot.complete == 0
    assert snapshot.metrics_text(1789000000.125) == (
        "crawl4ai_authoritative_task_count 3\n"
        "crawl4ai_direct_healthy_task_count 2\n"
        "crawl4ai_public_covered_task_count 1\n"
        "crawl4ai_coverage_complete 0\n"
        "crawl4ai_coverage_sample_timestamp_seconds 1789000000.125\n"
    )
    empty = observer.CoverageSnapshot(0, 0, 0)
    assert empty.complete == 0
    assert "crawl4ai_coverage_complete 0\n" in empty.metrics_text(1789000000.125)
    assert OBSERVER_COMPLETE.complete == 1
    one_task_short = rollout_policy.REPLICAS - 1
    assert observer.CoverageSnapshot(one_task_short, one_task_short, one_task_short).complete == 0
    with pytest.raises(ValueError, match="exceeds direct_healthy_count"):
        observer.CoverageSnapshot(3, 1, 2)


def test_public_coverage_values_are_redacted_aggregate_counts(tmp_path):
    diagnostic = observer.CoverageEpisodeLatch(tmp_path / "coverage-state").observe(
        OBSERVER_INCOMPLETE, OBSERVER_MISMATCH
    )

    assert diagnostic is not None
    assert [field.name for field in fields(observer.CoverageSnapshot)] == [
        "authoritative_task_count",
        "direct_healthy_count",
        "public_covered_count",
    ]
    assert [field.name for field in fields(observer.CoverageDiagnostic)] == [
        "authoritative_task_count",
        "direct_healthy_count",
        "public_covered_count",
        "networkdb_fdb_comparison",
    ]
    assert diagnostic.text() == (
        "crawl4ai coverage incomplete: "
        "authoritative_tasks=3 direct_healthy=2 public_covered=1 "
        "networkdb_fdb=destination_mismatch"
    )
    with pytest.raises(TypeError, match="integer"):
        observer.CoverageSnapshot("task-id", 0, 0)
    with pytest.raises(TypeError, match="CoverageSnapshot"):
        observer.CoverageEpisodeLatch(tmp_path / "other-state").observe(
            object(), OBSERVER_MISMATCH
        )
    with pytest.raises(TypeError, match="NetworkDbFdbComparison"):
        observer.CoverageEpisodeLatch(tmp_path / "other-state").observe(
            OBSERVER_INCOMPLETE, "10.0.1.38"
        )


def test_fdb_comparison_distinguishes_missing_and_wrong_destinations(monkeypatch):
    task = [{"ID": "task-1", "Node": "node-1"}]
    runtime = {"task-1": {"addresses": {"10.0.1.38"}}}
    data_path = {"task-1": "100.0.0.9"}

    def bridge(destination):
        return subprocess.CompletedProcess(
            [],
            0,
            ""
            if destination is None
            else f"02:42:0a:00:01:26 dst {destination} self permanent\n",
            "",
        )

    monkeypatch.setattr(
        observer.subprocess, "run", lambda *_args, **_kwargs: bridge(None)
    )
    assert (
        observer._fdb_comparison("network-id", task, runtime, data_path)
        is observer.NetworkDbFdbComparison.MISSING
    )
    monkeypatch.setattr(
        observer.subprocess, "run", lambda *_args, **_kwargs: bridge("100.0.0.8")
    )
    assert (
        observer._fdb_comparison("network-id", task, runtime, data_path)
        is observer.NetworkDbFdbComparison.DESTINATION_MISMATCH
    )
    monkeypatch.setattr(
        observer.subprocess, "run", lambda *_args, **_kwargs: bridge("100.0.0.9")
    )
    assert (
        observer._fdb_comparison("network-id", task, runtime, data_path)
        is observer.NetworkDbFdbComparison.CONSISTENT
    )
    monkeypatch.setattr(
        observer.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, "02:42:0a:00:01:26 malformed\n", ""
        ),
    )
    assert (
        observer._fdb_comparison("network-id", task, runtime, data_path)
        is observer.NetworkDbFdbComparison.INCONCLUSIVE
    )
