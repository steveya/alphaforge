from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from alphaforge.time.grids import build_grid_utc  # changed: use existing builder


class EventSource(ABC):
    """Produces event timestamps in UTC."""

    name: str

    @abstractmethod
    def events(
        self,
        ctx,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        entities: Optional[Sequence[str]] = None,
    ) -> pd.DatetimeIndex: ...


def _ensure_utc(ts: pd.DatetimeIndex | pd.Timestamp) -> pd.DatetimeIndex:
    if isinstance(ts, pd.Timestamp):
        ts = pd.DatetimeIndex([ts])
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _shift(ts: pd.DatetimeIndex, delay: pd.Timedelta) -> pd.DatetimeIndex:
    return (ts + delay).tz_convert("UTC")


@dataclass(frozen=True)
class SessionCloseEvents(EventSource):
    """Emit UTC session-close timestamps (optionally delayed)."""

    calendar: str
    delay: pd.Timedelta = pd.Timedelta(0)
    name: str = "session_close"

    def events(
        self,
        ctx,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        entities: Optional[Sequence[str]] = None,
    ) -> pd.DatetimeIndex:
        start_utc = _ensure_utc(start_utc)[0]
        end_utc = _ensure_utc(end_utc)[0]
        cal = ctx.calendars.get(self.calendar)
        if cal is None:
            raise KeyError(f"Calendar {self.calendar} not found in context")
        base = build_grid_utc(cal, start_utc, end_utc, "sessions")
        return _shift(pd.DatetimeIndex(base).tz_convert("UTC"), self.delay)


@dataclass(frozen=True)
class FixedIntervalEvents(EventSource):
    """Emit intraday timestamps at freq during sessions (optionally delayed)."""

    calendar: str
    freq: str  # e.g., "5min"
    delay: pd.Timedelta = pd.Timedelta(0)
    name: str = "fixed_interval"

    def events(
        self,
        ctx,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        entities: Optional[Sequence[str]] = None,
    ) -> pd.DatetimeIndex:
        start_utc = _ensure_utc(start_utc)[0]
        end_utc = _ensure_utc(end_utc)[0]
        cal = ctx.calendars.get(self.calendar)
        if cal is None:
            raise KeyError(f"Calendar {self.calendar} not found in context")
        base = build_grid_utc(cal, start_utc, end_utc, self.freq)
        return _shift(pd.DatetimeIndex(base).tz_convert("UTC"), self.delay)
