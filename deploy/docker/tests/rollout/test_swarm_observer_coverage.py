import subprocess
from dataclasses import fields
from pathlib import Path

import pytest

import swarm_observer as observer


OBSERVER_MISMATCH = observer.NetworkDbFdbComparison.DESTINATION_MISMATCH


OBSERVER_COMPLETE = observer.CoverageSnapshot(3, 3, 3)


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
    two_surviving_tasks = observer.CoverageSnapshot(2, 2, 2)
    assert two_surviving_tasks.complete == 0
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
    # The parsing logic is exercised with a known-resolved netns file, so the
    # bridge fdb output (not netns resolution) drives the assertions below.
    monkeypatch.setattr(
        observer,
        "_overlay_namespace",
        lambda _network_id: Path("/var/run/docker/netns/1-network-id"),
    )

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


def test_overlay_namespace_resolves_the_libnetwork_12_char_netns_filename(
    monkeypatch, tmp_path
):
    # Regression for swarm_observer's wrong-formula bug: libnetwork truncates
    # the whole "<epoch>-<network_id>" sandbox key to 12 characters, yielding
    # "<epoch>-<network_id[:10]>" (epoch normally "1"). The matching 14-char
    # file "1-<network_id[:12]>" the buggy code constructed can never exist.
    network_id = "0123456789abcdefcdef01"
    netns_dir = tmp_path / "netns"
    netns_dir.mkdir()
    (netns_dir / f"1-{network_id[:10]}").touch()
    monkeypatch.setattr(observer, "NETNS_DIR", netns_dir)

    namespace = observer._overlay_namespace(network_id)

    assert namespace == netns_dir / f"1-{network_id[:10]}"


def test_overlay_namespace_does_not_match_the_buggy_14_char_filename(
    monkeypatch, tmp_path
):
    # Even if someone were to leave a "1-<network_id[:12]>" (14-char) file on
    # disk, the suffix matcher must reject it: Docker would never create it,
    # and selecting it would re-introduce the original symptom (nsenter exits
    # 1 with "No such file or directory" because the real sandbox file is
    # the 12-char "<epoch>-<network_id[:10]>").
    network_id = "0123456789abcdefcdef01"
    netns_dir = tmp_path / "netns"
    netns_dir.mkdir()
    (netns_dir / f"1-{network_id[:12]}").touch()
    monkeypatch.setattr(observer, "NETNS_DIR", netns_dir)

    assert observer._overlay_namespace(network_id) is None


def test_overlay_namespace_picks_the_highest_epoch_candidate(monkeypatch, tmp_path):
    # libnetwork increments the sandbox epoch whenever the overlay network's
    # sandbox is reinitialized, so the file is renamed "2-<id[:10]>", "3-...",
    # etc. across daemon lifetimes. The matcher must take the highest epoch
    # numerically, not lexicographically (otherwise "10-..." is sorted before
    # "2-..." and the wrong, stale sandbox is selected).
    network_id = "0123456789abcdefcdef01"
    netns_dir = tmp_path / "netns"
    netns_dir.mkdir()
    for epoch in ("1", "2", "10"):
        (netns_dir / f"{epoch}-{network_id[:10]}").touch()
    monkeypatch.setattr(observer, "NETNS_DIR", netns_dir)

    namespace = observer._overlay_namespace(network_id)

    assert namespace is not None
    assert namespace.name == f"10-{network_id[:10]}"


def test_overlay_namespace_returns_none_when_no_candidate_matches(
    monkeypatch, tmp_path
):
    # No netns directory at all -> None. Files matching a different network
    # ID suffix -> None for the network ID we asked about.
    network_id = "0123456789abcdefcdef01"
    monkeypatch.setattr(observer, "NETNS_DIR", tmp_path / "missing-netns")

    assert observer._overlay_namespace(network_id) is None

    netns_dir = tmp_path / "netns"
    netns_dir.mkdir()
    (netns_dir / "1-deadbeefbe").touch()  # unrelated overlay network sandbox
    monkeypatch.setattr(observer, "NETNS_DIR", netns_dir)

    assert observer._overlay_namespace(network_id) is None


def test_fdb_comparison_invokes_nsenter_with_the_libnetwork_netns_filename(
    monkeypatch, tmp_path
):
    # End-to-end regression: with the 12-char "<epoch>-<network_id[:10]>" file
    # present, _fdb_comparison must invoke nsenter with --net=<that file> and
    # produce CONSISTENT. This is the path that previously always failed with
    # CalledProcessError because the code requested a non-existent 14-char
    # file.
    network_id = "0123456789abcdefcdef01"
    netns_dir = tmp_path / "netns"
    netns_dir.mkdir()
    (netns_dir / f"1-{network_id[:10]}").touch()
    monkeypatch.setattr(observer, "NETNS_DIR", netns_dir)

    invoked = {}

    def run(command, **_kwargs):
        invoked["command"] = command
        return subprocess.CompletedProcess(
            command, 0,
            "02:42:0a:00:01:26 dst 100.0.0.9 self permanent\n",
            "",
        )

    monkeypatch.setattr(observer.subprocess, "run", run)
    task = [{"ID": "task-1", "Node": "node-1"}]
    runtime = {"task-1": {"addresses": {"10.0.1.38"}}}
    data_path = {"task-1": "100.0.0.9"}

    comparison = observer._fdb_comparison(network_id, task, runtime, data_path)

    assert comparison is observer.NetworkDbFdbComparison.CONSISTENT
    assert invoked["command"] == [
        "nsenter",
        f"--net={netns_dir / f'1-{network_id[:10]}'}",
        "bridge",
        "fdb",
        "show",
        "dev",
        "vxlan0",
    ]


def test_fdb_comparison_returns_ns_enter_unavailable_when_netns_is_missing(
    monkeypatch, tmp_path
):
    # When the overlay sandbox netns file does not exist, _fdb_comparison
    # must return the distinct NS_ENTER_UNAVAILABLE outcome (not
    # INCONCLUSIVE) and never invoke nsenter — the previous name-construction
    # bug always landed here.
    monkeypatch.setattr(observer, "NETNS_DIR", tmp_path / "missing-netns")
    invoked = False

    def run(*_args, **_kwargs):
        nonlocal invoked
        invoked = True
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(observer.subprocess, "run", run)
    task = [{"ID": "task-1", "Node": "node-1"}]
    runtime = {"task-1": {"addresses": {"10.0.1.38"}}}
    data_path = {"task-1": "100.0.0.9"}

    comparison = observer._fdb_comparison(
        "0123456789abcdefcdef01", task, runtime, data_path
    )

    assert comparison is observer.NetworkDbFdbComparison.NS_ENTER_UNAVAILABLE
    assert comparison is not observer.NetworkDbFdbComparison.INCONCLUSIVE
    assert invoked is False


def test_fdb_comparison_returns_ns_enter_unavailable_when_nsenter_exits_nonzero(
    monkeypatch, tmp_path
):
    # nsenter's own failure modes (here: CalledProcessError, the symptom the
    # original wrong-formula bug always produced) must surface as
    # NS_ENTER_UNAVAILABLE rather than escape and be silently swallowed by
    # sample()'s old blanket `except Exception -> INCONCLUSIVE` clause.
    network_id = "0123456789abcdefcdef01"
    netns_dir = tmp_path / "netns"
    netns_dir.mkdir()
    (netns_dir / f"1-{network_id[:10]}").touch()
    monkeypatch.setattr(observer, "NETNS_DIR", netns_dir)

    def run(command, **_kwargs):
        raise subprocess.CalledProcessError(
            1, command,
            "nsenter: cannot open '/var/run/docker/netns/1-0123456789': "
            "No such file or directory",
        )

    monkeypatch.setattr(observer.subprocess, "run", run)
    task = [{"ID": "task-1", "Node": "node-1"}]
    runtime = {"task-1": {"addresses": {"10.0.1.38"}}}
    data_path = {"task-1": "100.0.0.9"}

    comparison = observer._fdb_comparison(network_id, task, runtime, data_path)

    assert comparison is observer.NetworkDbFdbComparison.NS_ENTER_UNAVAILABLE


def test_fdb_comparison_returns_ns_enter_unavailable_on_nsenter_timeout(
    monkeypatch, tmp_path
):
    network_id = "0123456789abcdefcdef01"
    netns_dir = tmp_path / "netns"
    netns_dir.mkdir()
    (netns_dir / f"1-{network_id[:10]}").touch()
    monkeypatch.setattr(observer, "NETNS_DIR", netns_dir)

    def run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="nsenter", timeout=15)

    monkeypatch.setattr(observer.subprocess, "run", run)
    task = [{"ID": "task-1", "Node": "node-1"}]
    runtime = {"task-1": {"addresses": {"10.0.1.38"}}}
    data_path = {"task-1": "100.0.0.9"}

    comparison = observer._fdb_comparison(network_id, task, runtime, data_path)

    assert comparison is observer.NetworkDbFdbComparison.NS_ENTER_UNAVAILABLE


def test_coverage_diagnostic_renders_ns_enter_unavailable_distinctly(tmp_path):
    # The operator-visible diagnostic line must distinguish a persistent
    # nsenter/netns failure from a genuine "could not measure" INCONCLUSIVE
    # so an observer cannot silently pin every incomplete episode at
    # networkdb_fdb=inconclusive while the FDB comparison is actually dead.
    diagnostic = observer.CoverageEpisodeLatch(tmp_path / "state").observe(
        observer.CoverageSnapshot(3, 2, 1),
        observer.NetworkDbFdbComparison.NS_ENTER_UNAVAILABLE,
    )

    assert diagnostic is not None
    assert diagnostic.text() == (
        "crawl4ai coverage incomplete: "
        "authoritative_tasks=3 direct_healthy=2 public_covered=1 "
        "networkdb_fdb=ns_enter_unavailable"
    )
    assert (
        observer.NetworkDbFdbComparison.NS_ENTER_UNAVAILABLE.value
        != observer.NetworkDbFdbComparison.INCONCLUSIVE.value
    )
