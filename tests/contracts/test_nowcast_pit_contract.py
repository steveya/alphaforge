from __future__ import annotations

import pandas as pd

from alphaforge import PITAccessor, RefSnapshotQuery
from alphaforge.store.duckdb_parquet import DuckDBParquetStore
from alphaforge.time.ref_period import RefPeriod


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def _sample_nowcast_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "CPI", "CPI"],
            "obs_date": [
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2024-12-31"),
            ],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-01-15", tz="UTC"),
                pd.Timestamp("2025-02-15", tz="UTC"),
            ],
            "value": [1.0, 1.1, 3.0, 3.1],
        }
    )


def test_nowcast_style_ref_queries_and_panel_builder_contract(tmp_path) -> None:
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_nowcast_df())

    snap = pit.snapshot_ref(
        RefSnapshotQuery(
            series_key="GDP",
            asof=pd.Timestamp("2025-03-01", tz="UTC"),
            start_ref="2024Q4",
            end_ref="2024Q4",
        )
    )
    panel = pit.build_snapshot_panel_long(
        [
            {
                "series_key": "GDP",
                "alias": "gdp",
                "start_ref": "2024Q4",
                "end_ref": "2024Q4",
                "freq": "Q",
            },
            {
                "series_key": "CPI",
                "alias": "cpi",
                "start_ref": "2024Q4",
                "end_ref": "2024Q4",
                "freq": "Q",
            },
        ],
        asof=pd.Timestamp("2025-03-01", tz="UTC"),
        align="quarter_end",
    )

    assert list(snap.index) == [RefPeriod.parse("2024Q4")]
    assert snap.loc[RefPeriod.parse("2024Q4")] == 1.1
    assert {"series_alias", "source_obs_date", "source_asof_utc", "value"} <= set(
        panel.columns
    )
    assert set(panel["series_alias"]) == {"gdp", "cpi"}
    assert panel["source_asof_utc"].notna().all()

