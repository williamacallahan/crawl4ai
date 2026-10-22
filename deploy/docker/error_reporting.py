"""Sentry enrollment for the Docker server processes."""

import os

import sentry_sdk
from sentry_sdk.scrubber import EventScrubber


def init_error_reporting() -> None:
    """Enroll this process in Sentry when the deployment provides a DSN.

    Both server processes (gunicorn and the crawl job worker) reach this through
    setup_logging. Only the production Dokploy app carries a DSN, so local and
    test runs report nothing. Recursive scrubbing reaches credentials nested in
    config dicts held by frame locals, which the default top-level scrubber skips.
    """
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return
    sentry_sdk.init(
        dsn=dsn,
        environment="production",
        release=f"crawl4ai@{os.environ['C4AI_GIT_SHA']}",
        event_scrubber=EventScrubber(recursive=True),
    )
