import pandas as pd
import pytest

from alphaforge.pit.accessor import PITAccessor
from alphaforge.pit.tasks import (
    first_vintage_snapshot,
    forward_fill_with_staleness,
    latest_vintage_snapshot,
    qoq,
    revision_deltas,
    revision_event_stream,
    revision_events,
    revision_stability,
    revision_volatility,
    snapshot_at_horizon,
    yoy,
)
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
            "value": [1.0, 1.2, 2.0, 2.5],
            "source": ["test"] * 4,
        }
    )


def test_vintage_selectors_and_revision_metrics(tmp_path):
    pit = _make_accessor(tmp_path)
    pit.upsert_pit_observations(_sample_df())

    first = first_vintage_snapshot(pit, "GDP")
    latest = latest_vintage_snapshot(pit, "GDP")

    assert first.loc[pd.Timestamp("2024-12-31", tz="UTC")] == 1.0
    assert latest.loc[pd.Timestamp("2024-12-31", tz="UTC")] == 1.2

    horizon = snapshot_at_horizon(
        pit,
        "GDP",
        horizon=pd.Timedelta(days=45),
    )
    assert horizon.loc[pd.Timestamp("2024-12-31", tz="UTC")] == 1.2

    deltas = revision_deltas(pit, "GDP", pd.Timestamp("2024-12-31", tz="UTC"))
    assert deltas.dropna().iloc[-1] == pytest.approx(0.2)

    events = revision_events(
        pit,
        "GDP",
        pd.Timestamp("2024-12-31", tz="UTC"),
        min_abs_change=0.1,
    )
    assert not events.empty

    stability = revision_stability(pit, "GDP")
    assert not stability.empty
    assert {"n_vintages", "total_abs_revision", "revision_std"}.issubset(stability.columns)

    event_stream = revision_event_stream(pit, "GDP", min_abs_change=0.05)
    assert not event_stream.empty
    assert {"obs_date", "asof_utc", "value", "delta"}.issubset(event_stream.columns)

    volatility = revision_volatility(pit, "GDP")
    assert not volatility.empty
    assert volatility.name == "GDP_revision_volatility"


def test_staleness_and_growth_helpers():
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-31", tz="UTC"), pd.Timestamp("2024-03-31", tz="UTC")]
    )
    s = pd.Series([100.0, 110.0], index=idx)

    target = pd.date_range("2024-01-31", "2024-04-30", freq="ME", tz="UTC")
    st = forward_fill_with_staleness(
        s,
        max_staleness=pd.Timedelta(days=45),
        target_index=target,
    )
    assert "is_stale" in st.columns
    assert {"source_obs_date", "age", "age_days"}.issubset(st.columns)

    monthly = pd.Series(
        [100, 105, 110, 120, 125, 130, 132, 136, 140, 145, 150, 155, 160],
        index=pd.date_range("2024-01-31", periods=13, freq="ME", tz="UTC"),
        dtype=float,
    )

    yoy_out = yoy(monthly)
    qoq_out = qoq(monthly)
    assert pd.isna(yoy_out.iloc[0])
    assert pd.isna(qoq_out.iloc[0])
