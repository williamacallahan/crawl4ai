"""Every workflow job runs on this fork's self-hosted runners.

GitHub-hosted minutes are not budgeted for this repository; a job that names
`ubuntu-latest` (or any runner list without `self-hosted`) would bill them.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).resolve().parents[2] / ".github" / "workflows").glob("*.yml"))


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda path: path.name)
def test_every_job_runs_self_hosted(workflow):
    jobs = yaml.safe_load(workflow.read_text())["jobs"]
    hosted = {
        name: job.get("runs-on")
        for name, job in jobs.items()
        if "uses" not in job
        and "self-hosted" not in (job.get("runs-on") if isinstance(job.get("runs-on"), list) else [job.get("runs-on")])
    }
    assert hosted == {}, f"{workflow.name} jobs not on self-hosted runners: {hosted}"


def test_workflows_found():
    assert WORKFLOWS, "no workflow files found; the guard would pass vacuously"
