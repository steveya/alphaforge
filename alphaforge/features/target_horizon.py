from __future__ import annotations

import pandas as pd
from typing import Literal, Optional


def next_in_grid(
    asof_utc: pd.Timestamp, grid_utc: pd.DatetimeIndex
) -> Optional[pd.Timestamp]:
    """Return the next timestamp strictly greater than asof_utc, or None."""
    pos = grid_utc.searchsorted(asof_utc, side="right")
    if pos >= len(grid_utc):
        return None
    return pd.Timestamp(grid_utc[pos]).tz_convert("UTC")


def compute_target_end(
    asof_utc: pd.Timestamp,
    grid_utc: pd.DatetimeIndex,
    horizon: Optional[pd.Timedelta] = None,
    horizon_mode: Optional[Literal["next_grid_step"]] = None,
) -> Optional[pd.Timestamp]:
    if horizon is not None and horizon_mode is not None:
        raise ValueError(
            "Specify either horizon (Timedelta) or horizon_mode, not both."
        )
    if horizon is not None:
        return (asof_utc + horizon).tz_convert("UTC")
    if horizon_mode == "next_grid_step":
        return next_in_grid(asof_utc, grid_utc)
    return None
