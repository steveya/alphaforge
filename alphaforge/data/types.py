"""Unified data-layer value objects.

FetchResult  — immutable result of a data fetch (PIT or non-PIT).
CacheManifest — metadata about a cache population event.
StaleDataWarning — warning emitted when cached data exceeds staleness threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd


class StaleDataWarning(UserWarning):
    """Emitted when cached data exceeds the caller's staleness threshold."""


@dataclass(frozen=True)
class CacheManifest:
    """Metadata about a cache population event."""

    dataset: str
    source: str
    entity_keys: list[str]
    asof_range: tuple[date, date]
    populated_at: datetime
    row_count: int


@dataclass(frozen=True)
class FetchResult:
    """Immutable result of a data fetch.

    Parameters
    ----------
    data : pd.DataFrame
        The fetched data.  For PIT data the frame has columns
        ``[obs_date, asof_utc, value]`` (and possibly others).
        For non-PIT data it has ``[obs_date, value]`` (or OHLCV columns).
    source : str
        Which source adapter provided the data (e.g. ``"fred"``, ``"cftc"``).
    dataset : str
        Logical dataset name (e.g. ``"gdp"``, ``"cot.tff"``).
    is_pit : bool
        Whether the data carries point-in-time revision history.
    cached_at : datetime | None
        When this data was stored in cache.  ``None`` means freshly fetched
        (not served from cache).
    """

    data: pd.DataFrame
    source: str
    dataset: str
    is_pit: bool
    cached_at: Optional[datetime]

    @property
    def cache_age(self) -> Optional[timedelta]:
        """Time since this data was cached, or ``None`` if freshly fetched."""
        if self.cached_at is None:
            return None
        return datetime.now(timezone.utc) - self.cached_at

    def as_timeseries(
        self,
        asof: Optional[date] = None,
    ) -> pd.Series:
        """Extract a single obs_date-indexed value Series.

        For **PIT** data: filters to the latest revision with
        ``asof_utc <= asof`` per ``obs_date``, then returns the ``value``
        column.  If *asof* is ``None``, returns the latest revision overall.

        For **non-PIT** data: ignores *asof* and returns the ``value`` column
        indexed by ``obs_date``.
        """
        df = self.data.copy()

        if self.is_pit and "asof_utc" in df.columns:
            if asof is not None:
                asof_ts = pd.Timestamp(asof)
                # Coerce asof_utc to comparable type
                asof_col = pd.to_datetime(df["asof_utc"])
                df = df[asof_col <= asof_ts]

            # Keep latest revision per obs_date
            df = df.sort_values("asof_utc").groupby("obs_date", as_index=False).last()

        df = df.sort_values("obs_date")
        series = df.set_index("obs_date")["value"]
        series.index.name = "obs_date"
        return series
