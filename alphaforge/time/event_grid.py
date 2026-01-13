from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class EventGrid:
    """Irregular decision timestamps (UTC tz-aware)."""

    timestamps_utc: pd.DatetimeIndex
    name: str = "events"


def _ensure_utc(dtidx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if dtidx.tz is None:
        dtidx = dtidx.tz_localize("UTC")
    else:
        dtidx = dtidx.tz_convert("UTC")
    return dtidx


def normalize_grid(
    grid: EventGrid,
    *,
    start_utc: Optional[pd.Timestamp] = None,
    end_utc: Optional[pd.Timestamp] = None,
) -> pd.DatetimeIndex:
    """
    Normalize an EventGrid to a sorted, unique, tz-aware UTC DatetimeIndex.
    Optionally clip to [start_utc, end_utc].
    """
    if not isinstance(grid, EventGrid):
        raise TypeError("normalize_grid expects EventGrid")

    ts = _ensure_utc(pd.DatetimeIndex(grid.timestamps_utc))
    ts = ts.sort_values().unique()

    if start_utc is not None:
        start_utc = _ensure_utc(pd.DatetimeIndex([start_utc]))[0]
        ts = ts[ts >= start_utc]
    if end_utc is not None:
        end_utc = _ensure_utc(pd.DatetimeIndex([end_utc]))[0]
        ts = ts[ts <= end_utc]

    return ts
