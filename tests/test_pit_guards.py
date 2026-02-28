import pandas as pd

from alphaforge.pit.guards import ReleaseLagPolicy, effective_asof, pit_leakage_report


def test_effective_asof_with_series_lag_cutoff_and_embargo():
    asof = pd.Timestamp("2025-03-15 06:00:00", tz="UTC")
    policy = ReleaseLagPolicy(
        default_lag=pd.Timedelta(days=1),
        per_series_lag={"GDP": pd.Timedelta(days=2)},
        cutoff_hour_utc=8,
        embargo_until_utc={"GDP": pd.Timestamp("2025-03-12 23:00:00", tz="UTC")},
    )

    eff = effective_asof(asof, "GDP", policy)
    assert eff == pd.Timestamp("2025-03-12 23:00:00", tz="UTC")


def test_pit_leakage_report_counts_violations_and_duplicates():
    df = pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "GDP"],
            "obs_date": [
                pd.Timestamp("2025-01-31", tz="UTC"),
                pd.Timestamp("2025-01-31", tz="UTC"),
                pd.Timestamp("2025-02-28", tz="UTC"),
            ],
            "asof_utc": [
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-02-01", tz="UTC"),
            ],
            "value": [1.0, 1.0, 2.0],
        }
    )

    report = pit_leakage_report(df)
    assert int(report.loc[0, "rows"]) == 3
    assert int(report.loc[0, "duplicate_key_rows"]) == 1
    assert int(report.loc[0, "future_rows"]) == 1
