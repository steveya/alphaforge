import pandas as pd
import pytest

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.exceptions import PITContractError
from alphaforge.store.duckdb_parquet import DuckDBParquetStore


def _make_accessor(tmp_path) -> PITAccessor:
    store = DuckDBParquetStore(root=str(tmp_path))
    return PITAccessor(store.conn())


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "CPI", "CPI", "GDP", "CPI"],
            "obs_date": [
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
                pd.Timestamp("2024-03-31"),
                pd.Timestamp("2024-03-31"),
            ],
            "asof_utc": [
                pd.Timestamp("2024-03-10", tz="UTC"),
                pd.Timestamp("2024-03-10", tz="UTC"),
                pd.Timestamp("2024-03-05", tz="UTC"),
                pd.Timestamp("2024-03-05", tz="UTC"),
                pd.Timestamp("2024-04-20", tz="UTC"),
                pd.Timestamp("2024-04-20", tz="UTC"),
            ],
            "value": [3.0, 3.5, 1.0, 1.1, 4.0, 1.2],
        }
    )


def test_list_union_vintages_event_and_calendar(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    event = pit.list_union_vintages(["GDP", "CPI"], mode="event")
    assert list(event) == [
        pd.Timestamp("2024-03-05", tz="UTC"),
        pd.Timestamp("2024-03-10", tz="UTC"),
        pd.Timestamp("2024-04-20", tz="UTC"),
    ]

    cal = pit.list_union_vintages(
        ["GDP", "CPI"],
        start=pd.Timestamp("2024-03-01", tz="UTC"),
        end=pd.Timestamp("2024-03-03", tz="UTC"),
        mode="calendar",
        calendar_freq="D",
    )
    assert list(cal) == [
        pd.Timestamp("2024-03-01", tz="UTC"),
        pd.Timestamp("2024-03-02", tz="UTC"),
        pd.Timestamp("2024-03-03", tz="UTC"),
    ]

    with pytest.raises(PITContractError, match="mode"):
        pit.list_union_vintages(["GDP"], mode="bad")


def test_build_snapshot_panel_alignment_and_policy(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    panel_month = pit.build_snapshot_panel(
        [
            {"series_key": "GDP", "alias": "gdp"},
            {"series_key": "CPI", "alias": "cpi", "release_policy": "latest"},
        ],
        asof=pd.Timestamp("2024-04-30", tz="UTC"),
        align="month_end",
        join="outer",
    )
    assert {"gdp", "cpi"}.issubset(panel_month.columns)

    panel_quarter = pit.build_snapshot_panel(
        [
            {"series_key": "GDP", "alias": "gdp", "release_policy": "latest"},
        ],
        asof=pd.Timestamp("2024-04-30", tz="UTC"),
        align="quarter_end",
        join="outer",
    )
    assert list(panel_quarter.index) == [pd.Timestamp("2024-03-31", tz="UTC")]

    panel_first = pit.build_snapshot_panel(
        [
            {"series_key": "GDP", "alias": "gdp_first", "release_policy": "first"},
        ],
        asof=pd.Timestamp("2024-04-30", tz="UTC"),
    )
    assert float(panel_first.iloc[-1]["gdp_first"]) == pytest.approx(4.0)


def test_build_snapshot_panel_respects_asof_cut(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    panel = pit.build_snapshot_panel(
        [{"series_key": "GDP", "alias": "gdp"}],
        asof=pd.Timestamp("2024-03-15", tz="UTC"),
    )
    assert panel.index.max() == pd.Timestamp("2024-02-29", tz="UTC")
