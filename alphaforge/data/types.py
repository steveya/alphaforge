"""Unified data-layer value objects.

FetchResult  — frozen fetch metadata with a caller-owned DataFrame.
CacheManifest — metadata about a cache population event.
StaleDataWarning — warning emitted when cached data exceeds staleness threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from .contracts import AvailabilityBasis, AvailabilityPrecision, ResultKind


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
class FetchFailure:
    """A failed entity acquisition in an explicitly allowed partial result."""

    entity: str | None
    message: str
    error_type: str = ""


@dataclass(frozen=True)
class AcquisitionMetadata:
    """Requested slice and cache origin for one series or bulk acquisition."""

    entity: str | None = None
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    asof: pd.Timestamp | None = None
    cached_at: datetime | None = None
    cache_variant: str = ""
    cache_namespace: str | None = None


@dataclass(frozen=True)
class FetchResult:
    """Frozen result metadata with a caller-owned DataFrame.

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
        Legacy flag for availability-aware data; it does not prove that a
        complete revision history or original release times are present.
    cached_at : datetime | None
        Oldest cache timestamp among acquisitions, or ``None`` if all are fresh.
        Inspect ``acquisitions`` for mixed fresh/cached results.
    """

    data: pd.DataFrame
    source: str
    dataset: str
    is_pit: bool
    cached_at: Optional[datetime]
    result_kind: ResultKind | None = None
    availability_precision: AvailabilityPrecision = "none"
    requested_entities: tuple[str, ...] | None = None
    failures: tuple[FetchFailure, ...] = ()
    acquisitions: tuple[AcquisitionMetadata, ...] = ()
    snapshot_asof: pd.Timestamp | None = None
    availability_basis: AvailabilityBasis = "none"
    adapter_name: str | None = None

    def __post_init__(self) -> None:
        if self.acquisitions:
            cached_times = [
                pd.to_datetime(item.cached_at, utc=True).to_pydatetime()
                for item in self.acquisitions if item.cached_at is not None
            ]
            object.__setattr__(self, "cached_at", min(cached_times) if cached_times else None)
        elif self.cached_at is not None:
            object.__setattr__(self, "cached_at", pd.to_datetime(self.cached_at, utc=True).to_pydatetime())
        if self.snapshot_asof is not None:
            object.__setattr__(self, "snapshot_asof", pd.to_datetime(self.snapshot_asof, utc=True))

    @property
    def complete(self) -> bool:
        """Whether every attempted acquisition succeeded (empty data can be complete)."""
        return not self.failures

    @property
    def cache_age(self) -> Optional[timedelta]:
        """Time since this data was cached, or ``None`` if freshly fetched."""
        if self.cached_at is None:
            return None
        return datetime.now(timezone.utc) - self.cached_at

    def as_timeseries(
        self,
        asof: date | datetime | pd.Timestamp | None = None,
        *,
        entity: str | None = None,
        value_column: str = "value",
    ) -> pd.Series:
        """Extract a single obs_date-indexed value Series.

        For **PIT** data: filters to the latest revision with
        ``asof_utc <= asof`` per ``obs_date``, then returns the ``value``
        column.  If *asof* is ``None``, returns the latest revision overall.

        Select ``entity`` when more than one series is present, and use
        ``value_column`` for market fields such as close or volume. Selected
        snapshots can only serve their own cutoff (UTC day for date precision).
        Explicit observation
        results reject as-of filtering; legacy non-PIT results retain their
        historical behavior of ignoring *asof*.
        """
        df = self.data.copy()
        missing = {"obs_date", value_column} - set(df.columns)
        if missing:
            raise ValueError(f"Timeseries result is missing columns: {sorted(missing)}")
        if "series_key" in df:
            if df["series_key"].isna().any():
                raise ValueError("Timeseries result contains missing entity keys.")
            if entity is not None:
                df = df[df["series_key"] == entity]
            elif df["series_key"].nunique() > 1:
                raise ValueError("Specify entity= for a result containing multiple series.")
        elif entity is not None:
            raise ValueError("Selecting entity requires a series_key column.")

        if self.result_kind == "observations" and asof is not None:
            raise ValueError("Observation results do not support asof filtering.")
        if self.result_kind == "snapshot" and asof is not None:
            cutoff = pd.to_datetime(asof, utc=True)
            snapshot = self.snapshot_asof
            if self.availability_precision == "date":
                cutoff = cutoff.normalize()
                snapshot = snapshot.normalize() if snapshot is not None else None
            if snapshot is None or cutoff != snapshot:
                raise ValueError("A selected snapshot can only serve its own snapshot cutoff.")

        if self.is_pit:
            if "asof_utc" not in df:
                raise ValueError("Availability-aware timeseries require an asof_utc column.")
            df["asof_utc"] = pd.to_datetime(df["asof_utc"], utc=True)
            if asof is not None:
                asof_ts = pd.to_datetime(asof, utc=True)
                available = df["asof_utc"]
                if self.availability_precision == "date":
                    available = available.dt.normalize()
                    asof_ts = asof_ts.normalize()
                df = df[available <= asof_ts]

            # Select entire latest rows: groupby.last() would skip null revisions.
            df = df.sort_values("asof_utc", kind="stable").drop_duplicates("obs_date", keep="last")

        df = df.sort_values("obs_date")
        if df["obs_date"].duplicated().any():
            raise ValueError("Timeseries result has multiple observations per date; select one entity.")
        series = df.set_index("obs_date")[value_column]
        series.index.name = "obs_date"
        return series
