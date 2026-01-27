import pandas as pd

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.ref_entity import make_ref_entity_id, parse_ref_entity_id
from alphaforge.time.ref_period import RefPeriod
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def _sample_df() -> pd.DataFrame:
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
            "source": ["alfred"] * 4,
        }
    )


def test_upsert_idempotency(tmp_path):
    pit = _make_accessor(tmp_path)
    df = _sample_df()
    pit.upsert_pit_observations(df)
    pit.upsert_pit_observations(df)
    count = pit.conn.execute(
        "SELECT COUNT(*) FROM pit_observations WHERE series_key = ?",
        ["GDP"],
    ).fetchone()[0]
    assert count == len(df)


def test_snapshot_selection(tmp_path):
    pit = _make_accessor(tmp_path)
    df = _sample_df()
    pit.upsert_pit_observations(df)

    snap_early = pit.get_snapshot("GDP", pd.Timestamp("2025-01-15", tz="UTC"))
    assert snap_early.loc[pd.Timestamp("2024-12-31", tz="UTC")] == 1.0

    snap_late = pit.get_snapshot("GDP", pd.Timestamp("2025-03-01", tz="UTC"))
    assert snap_late.loc[pd.Timestamp("2024-12-31", tz="UTC")] == 1.1


def test_snapshot_start_end_filtering(tmp_path):
    pit = _make_accessor(tmp_path)
    df = _sample_df()
    pit.upsert_pit_observations(df)
    snap = pit.get_snapshot(
        "GDP",
        pd.Timestamp("2025-06-01", tz="UTC"),
        start=pd.Timestamp("2025-03-31", tz="UTC"),
        end=pd.Timestamp("2025-03-31", tz="UTC"),
    )
    assert list(snap.index) == [pd.Timestamp("2025-03-31", tz="UTC")]
    assert snap.iloc[0] == 2.1


def test_revision_timeline_ordering(tmp_path):
    pit = _make_accessor(tmp_path)
    df = _sample_df()
    pit.upsert_pit_observations(df)
    timeline = pit.get_revision_timeline("GDP", pd.Timestamp("2024-12-31", tz="UTC"))
    assert list(timeline.index) == [
        pd.Timestamp("2025-01-10", tz="UTC"),
        pd.Timestamp("2025-02-10", tz="UTC"),
    ]
    assert list(timeline.values) == [1.0, 1.1]


def test_multiple_series_key_isolation(tmp_path):
    pit = _make_accessor(tmp_path)
    df = _sample_df()
    extra = pd.DataFrame(
        {
            "series_key": ["CPI"],
            "obs_date": [pd.Timestamp("2024-12-31")],
            "asof_utc": [pd.Timestamp("2025-01-20", tz="UTC")],
            "value": [5.0],
            "source": ["alfred"],
        }
    )
    pit.upsert_pit_observations(pd.concat([df, extra], ignore_index=True))
    snap = pit.get_snapshot("CPI", pd.Timestamp("2025-02-01", tz="UTC"))
    assert list(snap.index) == [pd.Timestamp("2024-12-31", tz="UTC")]
    assert snap.iloc[0] == 5.0


def test_revision_timeline_ref_matches_obs_date(tmp_path):
    pit = _make_accessor(tmp_path)
    df = _sample_df()
    pit.upsert_pit_observations(df)
    by_ref = pit.get_revision_timeline_ref("GDP", "2024Q4")
    by_obs = pit.get_revision_timeline("GDP", pd.Timestamp("2024-12-31", tz="UTC"))
    pd.testing.assert_series_equal(by_ref, by_obs)


def test_snapshot_ref_range_bounds(tmp_path):
    pit = _make_accessor(tmp_path)
    df = _sample_df()
    pit.upsert_pit_observations(df)
    snap = pit.get_snapshot_ref(
        "GDP",
        pd.Timestamp("2025-06-01", tz="UTC"),
        start_ref="2024Q4",
        end_ref="2024Q4",
    )
    assert list(snap.index) == [pd.Timestamp("2024-12-31", tz="UTC")]

    snap_range = pit.get_snapshot_ref(
        "GDP",
        pd.Timestamp("2025-06-01", tz="UTC"),
        start_ref="2024Q4",
        end_ref="2025Q1",
    )
    assert list(snap_range.index) == [
        pd.Timestamp("2024-12-31", tz="UTC"),
        pd.Timestamp("2025-03-31", tz="UTC"),
    ]


def test_ref_entity_id_helpers():
    ref = RefPeriod.parse("2024Q4")
    entity_id = make_ref_entity_id("GDP", ref)
    assert entity_id == "GDP|2024Q4"
    series_key, parsed = parse_ref_entity_id(entity_id)
    assert series_key == "GDP"
    assert parsed == ref
