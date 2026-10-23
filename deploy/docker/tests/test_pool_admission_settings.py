"""Typed configuration for the replica-wide page admission budget."""

from pathlib import Path

import pytest
import yaml

import utils


@pytest.mark.parametrize("config", [{}, {"crawler": {}}, {"crawler": {"pool": {}}}])
def test_missing_pool_admission_settings_use_defaults(config):
    settings = utils.PoolAdmissionSettings.from_config(config)
    assert settings.replica_max_pages == 8
    assert settings.admission_wait_seconds == 2.0


def test_shipped_and_fallback_pool_admission_settings_match():
    shipped = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "config.yml").read_text()
    )
    for config in (shipped, utils.DEFAULT_CONFIG):
        pool = config["crawler"]["pool"]
        assert pool["replica_max_pages"] == 8
        assert pool["admission_wait_seconds"] == 2.0
        assert utils.PoolAdmissionSettings.from_config(config) == utils.PoolAdmissionSettings()


@pytest.mark.parametrize("value", [True, False, "8", 8.0, None, 0, -1])
def test_replica_page_budget_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match=r"crawler\.pool\.replica_max_pages"):
        utils.PoolAdmissionSettings.from_config({"crawler": {"pool": {"replica_max_pages": value}}})


@pytest.mark.parametrize(
    "value", [True, False, "2", None, -0.1, float("inf"), float("-inf"), float("nan")]
)
def test_admission_wait_requires_a_finite_nonnegative_number(value):
    with pytest.raises(ValueError, match=r"crawler\.pool\.admission_wait_seconds"):
        utils.PoolAdmissionSettings.from_config(
            {"crawler": {"pool": {"admission_wait_seconds": value}}}
        )


@pytest.mark.parametrize("budget,wait", [(1, 0), (8, 0.25), (16, 3)])
def test_custom_budget_and_wait_include_fail_fast(budget, wait):
    settings = utils.PoolAdmissionSettings.from_config(
        {"crawler": {"pool": {"replica_max_pages": budget, "admission_wait_seconds": wait}}}
    )
    assert settings.replica_max_pages == budget
    assert settings.admission_wait_seconds == wait
    assert isinstance(settings.replica_max_pages, int)
    assert isinstance(settings.admission_wait_seconds, float)

