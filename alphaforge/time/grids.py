from dataclasses import dataclass
from typing import Sequence
import pandas as pd

from .calendar import TradingCalendar


@dataclass(frozen=True)
class Grid:
    name: str
    index: pd.DatetimeIndex


@dataclass(frozen=True)
class SessionGrid(Grid):
    calendar: str = "XNYS"


@dataclass(frozen=True)
class NativeGrid(Grid):
    table: str = ""


@dataclass(frozen=True)
class EventGrid(Grid):
    sources: Sequence[str] = ()
    tables: Sequence[str] = ()


def build_grid_utc(
    cal: TradingCalendar, start_utc: pd.Timestamp, end_utc: pd.Timestamp, grid: str
) -> pd.DatetimeIndex:
    """Build evaluation grid as tz-aware UTC DatetimeIndex.

    Supported grid forms:
      - "B" or "sessions" or "daily": emit session close timestamps (UTC-aware)
      - "{n}min": emit trading minutes during sessions at given frequency (e.g., "5min")

    Returns tz-aware UTC DatetimeIndex sorted and unique.
    """
    s = pd.Timestamp(start_utc)
    e = pd.Timestamp(end_utc)
    if s.tzinfo is None:
        s = s.tz_localize("UTC")
    if e.tzinfo is None:
        e = e.tz_localize("UTC")

    g = grid.lower() if isinstance(grid, str) else grid

    if g in ("b", "sessions", "daily"):
        # generate business dates between start and end (naive dates)
        dates = pd.bdate_range(start=s.date(), end=e.date())
        closes = []
        for d in dates:
            # session_close_utc accepts date-like and will coerce to UTC-aware
            close = cal.session_close_utc(d)
            closes.append(close)
        idx = pd.DatetimeIndex(closes)
        return idx.sort_values()

    # intraday frequency like '5min'
    if isinstance(g, str) and g.endswith("min"):
        minutes = cal.trading_minutes_utc(s, e, freq=g)
        return minutes

    raise ValueError(f"Unsupported grid spec: {grid}")
