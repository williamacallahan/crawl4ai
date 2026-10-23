"""Invalid requests cannot pin browser-config strings in monitor history."""

import hashlib
import json

import pytest

from monitor import MonitorStats


@pytest.mark.asyncio
async def test_monitor_retains_only_server_derived_signatures():
    stats = MonitorStats(redis=None)
    for index in range(3):
        signature = f"invalid-{index}-" + "x" * (9 * 1024 * 1024)
        expected = hashlib.sha256(signature.encode()).hexdigest()[:16]
        await stats.track_request_start(str(index), "/crawl", "https://example.com", {"sig": signature})
        assert stats.get_active_requests()[0]["config_sig"] == expected
        await stats.track_request_end(str(index), success=False, error="invalid config", status_code=422)
        assert stats.get_completed_requests()[-1]["config_sig"] == expected
    assert len(json.dumps(stats.get_completed_requests())) < 4096
    await stats.track_request_start("bad-type", "/crawl", "https://example.com", {"sig": {"large": "x" * 10000}})
    assert stats.get_active_requests()[0]["config_sig"] == "default"
