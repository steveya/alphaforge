import pandas as pd

from alphaforge import RefRevisionQuery as TopLevelRefRevisionQuery
from alphaforge import RefSnapshotQuery as TopLevelRefSnapshotQuery
from alphaforge.pit import PITAccessor, RefRevisionQuery, RefSnapshotQuery
from alphaforge.pit.ref_entity import make_ref_entity_id
from alphaforge.store.duckdb_parquet import DuckDBParquetStore
from alphaforge.time.ref_period import RefFreq, RefPeriod


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def _sample_quarterly_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "GDP", "GDP"],
            "obs_date": [
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2025-03-31"),
                pd.Timestamp("2025-03-31"),
            ],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-04-10", tz="UTC"),
                pd.Timestamp("2025-05-10", tz="UTC"),
            ],
            "value": [1.0, 1.1, 2.0, 2.1],
        }
    )


def _sample_monthly_start_anchor_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["CPI", "CPI"],
            "obs_date": [
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-02-01"),
            ],
            "asof_utc": [
                pd.Timestamp("2025-01-15", tz="UTC"),
                pd.Timestamp("2025-02-15", tz="UTC"),
            ],
            "value": [3.0, 3.1],
        }
    )


def test_public_ref_query_exports_are_canonical() -> None:
    assert TopLevelRefSnapshotQuery is RefSnapshotQuery
    assert TopLevelRefRevisionQuery is RefRevisionQuery


def test_snapshot_ref_query_returns_ref_period_index(tmp_path) -> None:
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_quarterly_df())

    snap = pit.snapshot_ref(
        RefSnapshotQuery(
            series_key="GDP",
            asof=pd.Timestamp("2025-06-01", tz="UTC"),
            start_ref="2024Q4",
            end_ref=pd.Period("2025Q1", freq="Q"),
        )
    )

    assert snap.index.name == "ref_period"
    assert list(snap.index) == [RefPeriod.parse("2024Q4"), RefPeriod.parse("2025Q1")]
    assert snap.loc[RefPeriod.parse("2024Q4")] == 1.1
    assert snap.loc[RefPeriod.parse("2025Q1")] == 2.1


def test_snapshot_ref_query_accepts_explicit_obs_date_anchor(tmp_path) -> None:
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_monthly_start_anchor_df())

    snap = pit.snapshot_ref(
        RefSnapshotQuery(
            series_key="CPI",
            asof=pd.Timestamp("2025-03-01", tz="UTC"),
            start_ref="2025-01-01",
            end_ref="2025-02-01",
            freq=RefFreq.M,
            obs_date_anchor="start",
        )
    )

    assert list(snap.index) == [RefPeriod.parse("2025-01"), RefPeriod.parse("2025-02")]
    assert snap.loc[RefPeriod.parse("2025-01")] == 3.0
    assert snap.loc[RefPeriod.parse("2025-02")] == 3.1


def test_revisions_ref_query_accepts_mapping_and_sets_ref_entity_name(tmp_path) -> None:
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_quarterly_df())

    timeline = pit.revisions_ref(
        {
            "series_key": "GDP",
            "ref": pd.Period("2024Q4", freq="Q"),
            "end_asof": pd.Timestamp("2025-02-15", tz="UTC"),
        }
    )

    assert list(timeline.index) == [
        pd.Timestamp("2025-01-10", tz="UTC"),
        pd.Timestamp("2025-02-10", tz="UTC"),
    ]
    assert list(timeline.values) == [1.0, 1.1]
    assert timeline.name == make_ref_entity_id("GDP", RefPeriod.parse("2024Q4"))
