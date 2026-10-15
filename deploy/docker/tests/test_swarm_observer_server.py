import http.client
from threading import Thread

import pytest

import swarm_observer as observer
import swarm_observer_server as observer_server

OBSERVER_COMPLETE = observer.CoverageSnapshot(3, 3, 3)


def test_metrics_endpoint_exposes_only_the_fixed_contract(tmp_path):
    path = tmp_path / "metrics"
    path.write_text(OBSERVER_COMPLETE.metrics_text(1789000000.125))
    metrics = observer_server.MetricsFile(path, 60)
    server = observer_server.http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), observer_server.handler(metrics)
    )
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        connection.request("GET", "/metrics")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read().decode() == OBSERVER_COMPLETE.metrics_text(1789000000.125)
        connection.request("GET", "/tasks")
        assert connection.getresponse().status == 404
        connection.request("POST", "/metrics")
        assert connection.getresponse().status == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_metrics_endpoint_fails_closed_when_the_sampler_is_stale(monkeypatch, tmp_path):
    path = tmp_path / "metrics"
    path.write_text(OBSERVER_COMPLETE.metrics_text(1789000000.125))
    monkeypatch.setattr(observer_server.time, "time", lambda: path.stat().st_mtime + 61)

    with pytest.raises(TimeoutError, match="stale"):
        observer_server.MetricsFile(path, 60).read()
