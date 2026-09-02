"""Unprivileged HTTP surface for the root sampler's bounded metrics file."""

from __future__ import annotations

import http.server
import os
import time
import urllib.parse
from http import HTTPStatus
from pathlib import Path


DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 9476
DEFAULT_METRICS_PATH = "/var/lib/crawl4ai-swarm-observer/metrics.prom"
DEFAULT_MAX_AGE_SECONDS = 60.0
METRIC_NAME = (
    "crawl4ai_authoritative_task_count",
    "crawl4ai_direct_healthy_task_count",
    "crawl4ai_public_covered_task_count",
    "crawl4ai_coverage_complete",
)


class MetricsFile:
    def __init__(self, path: str | Path, max_age_seconds: float) -> None:
        self._path = Path(path)
        self._max_age_seconds = max_age_seconds

    def read(self) -> bytes:
        with self._path.open() as source:
            if time.time() - os.fstat(source.fileno()).st_mtime > self._max_age_seconds:
                raise TimeoutError("coverage metrics are stale")
            text = source.read()
        line = text.splitlines()
        if len(line) != len(METRIC_NAME) or any(
            not sample.startswith(f"{name} ") or "{" in sample
            for name, sample in zip(METRIC_NAME, line, strict=True)
        ):
            raise ValueError("coverage metrics contract is invalid")
        return ("\n".join(line) + "\n").encode()


def handler(metrics: MetricsFile) -> type[http.server.BaseHTTPRequestHandler]:
    class MetricsHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urllib.parse.urlsplit(self.path).path != "/metrics":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                body = metrics.read()
            except (OSError, TimeoutError, ValueError):
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_arguments: object) -> None:
            return

    return MetricsHandler


def serve() -> None:
    bind = os.environ.get("CRAWL4AI_OBSERVER_BIND", DEFAULT_BIND)
    port = int(os.environ.get("CRAWL4AI_OBSERVER_PORT", DEFAULT_PORT))
    metrics = MetricsFile(
        os.environ.get("CRAWL4AI_OBSERVER_METRICS_PATH", DEFAULT_METRICS_PATH),
        DEFAULT_MAX_AGE_SECONDS,
    )
    http.server.ThreadingHTTPServer((bind, port), handler(metrics)).serve_forever()


if __name__ == "__main__":
    serve()
