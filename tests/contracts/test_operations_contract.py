from __future__ import annotations

import pandas as pd

from alphaforge.data.public_web.archive import (
    discover_archive_fetches,
    iter_yearly_archive_fetches,
)
from alphaforge.pipeline.health import (
    SourceHealthPolicy,
    assess_source_health,
    build_health_report,
)
from alphaforge.time.release_rules import FixedLagMonths


def test_operations_contract_for_health_and_archive_planning() -> None:
    policy = SourceHealthPolicy(
        expected_cadence=pd.Timedelta(days=31),
        release_rule=FixedLagMonths(months=2),
        grace_period=pd.Timedelta(days=2),
        stale_threshold=pd.Timedelta(days=14),
        dead_threshold=pd.Timedelta(days=42),
        weight_decay_half_life=pd.Timedelta(days=7),
    )
    status = assess_source_health(
        "macro",
        latest_obs_date=pd.Timestamp("2025-01-31", tz="UTC"),
        asof=pd.Timestamp("2025-04-05", tz="UTC"),
        policy=policy,
    )
    report = build_health_report({"macro": status})

    planned = discover_archive_fetches(
        '<a href="/files/report_2021.zip?download=1">report</a>',
        base_url="https://example.com/archive/index.html",
        suffixes=[".zip"],
        years=[2021],
        fallback_artifact_prefix="example",
    )
    yearly = iter_yearly_archive_fetches(
        start_year=2009,
        end_year=2012,
        url_template="https://example.com/data_{year}.zip",
        first_year=2006,
        yearly_first_year=2010,
        historical_url="https://example.com/historical.zip",
        historical_last_year=2011,
        fallback_artifact_prefix="example",
    )

    assert report.loc[0, "source_name"] == "macro"
    assert report.loc[0, "expected_next"] == pd.Timestamp("2025-04-01", tz="UTC")
    assert report.loc[0, "overdue_days"] == 4.0
    assert planned[0].artifact_name == "report_2021.zip"
    assert planned[0].year == 2021
    assert yearly[-1].artifact_name == "data_2012.zip"
    assert yearly[-1].year == 2012

