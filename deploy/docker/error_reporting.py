"""Sentry enrollment for the Docker server processes."""

import os

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration


def init_error_reporting() -> None:
    """Enroll this process in Sentry when the deployment provides a DSN.

    Both server processes (gunicorn and the crawl job worker) reach this through
    setup_logging. Only the production Dokploy app carries a DSN, so local and
    test runs report nothing.

    Events come only from ERROR logs. A 5xx response is not captured on its own:
    the server's 500 handler and exc_info logs already report real defects, while
    502/503/504 are target-site failures and designed backpressure. Frame locals
    and request bodies stay out because they carry caller webhook headers and the
    LLM key under names the key-based scrubber does not match.
    """
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return
    sentry_sdk.init(
        dsn=dsn,
        environment="production",
        release=f"crawl4ai@{os.environ['C4AI_GIT_SHA']}",
        include_local_variables=False,
        max_request_body_size="never",
        integrations=[
            StarletteIntegration(failed_request_status_codes=set()),
            FastApiIntegration(failed_request_status_codes=set()),
        ],
    )
