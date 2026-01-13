from __future__ import annotations

import pandas as pd

from ..time.align import AlignedPanel, AvailabilityState


def assert_grid_is_monotone(grid_utc: pd.DatetimeIndex) -> None:
    if grid_utc.tz is None:
        raise AssertionError("grid_utc must be tz-aware UTC")
    if not grid_utc.is_monotonic_increasing:
        raise AssertionError("grid_utc must be sorted (monotone increasing)")
    if grid_utc.has_duplicates:
        raise AssertionError("grid_utc must be unique")


def assert_no_future_data(df: pd.DataFrame, asof_col: str, ts_col: str) -> None:
    """
    Ensure PIT correctness: for any row, ts_col <= asof_col.
    Assumes columns are tz-aware UTC.
    """
    if df.empty:
        return
    if df[asof_col].dt.tz is None or df[ts_col].dt.tz is None:
        raise AssertionError("asof_col and ts_col must be tz-aware UTC")
    bad = df[df[ts_col] > df[asof_col]]
    if not bad.empty:
        raise AssertionError("Found future data relative to asof_utc (PIT violation).")


def leakage_audit_aligned(aligned: AlignedPanel) -> pd.DataFrame:
    val = aligned.value.df
    av = aligned.availability.df
    mask = av == AvailabilityState.NOT_YET_RELEASED.value
    viol = mask & val.notna()
    return pd.DataFrame(
        [
            {
                "not_yet_released_value_cells": int(viol.sum().sum()),
                "note": "NOT_YET_RELEASED requires vintage-aware DataSource or release calendars; MVP reserves the label.",
            }
        ]
    )
