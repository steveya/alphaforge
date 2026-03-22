"""Data models for PIT adapter observations and series metadata.

These types define the adapter-level contract for point-in-time data:

- :class:`PITObservation` — a single vintage observation from a data source.
- :class:`SeriesMetadata` — metadata describing a macro series and its PIT mode.
- :data:`PITMode` — enumeration of PIT access modes.
- :func:`create_pit_dataframe` — convert observations to a canonical DataFrame.
- :func:`create_wide_view` — pivot a single-asof DataFrame to wide format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

if TYPE_CHECKING:
    from alphaforge.pit.release_rules import ReleaseRule

PITMode = Literal["ALFRED_REALTIME", "DISCRETE_VINTAGES_SNAP", "NO_PIT"]


@dataclass
class SeriesMetadata:
    """Metadata for a macro series."""

    series_key: str
    country: str
    source: str
    source_series_id: str
    frequency: str
    pit_mode: PITMode
    seasonal_adjustment: str | None = None
    units: str | None = None
    description: str | None = None
    transforms: list | None = None
    adapter: str | None = None
    obs_date_anchor: Literal["start", "end"] | None = None
    publication_lag_months: int | None = None
    release_rule: ReleaseRule | None = None
    blocks: dict[str, int] = field(default_factory=dict)


@dataclass
class PITObservation:
    """A single point-in-time observation."""

    series_key: str
    source: str
    source_series_id: str
    asof_date: date
    vintage_date: date
    obs_date: date
    value: float
    frequency: str
    value_raw: str | None = None
    units: str | None = None
    seasonal_adjustment: str | None = None
    realtime_start: date | None = None
    realtime_end: date | None = None
    ingested_at: datetime | None = None
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "series_key": self.series_key,
            "source": self.source,
            "source_series_id": self.source_series_id,
            "asof_date": self.asof_date,
            "vintage_date": self.vintage_date,
            "obs_date": self.obs_date,
            "value": self.value,
            "value_raw": self.value_raw,
            "frequency": self.frequency,
            "units": self.units,
            "seasonal_adjustment": self.seasonal_adjustment,
            "realtime_start": self.realtime_start,
            "realtime_end": self.realtime_end,
            "ingested_at": self.ingested_at,
            "provenance": self.provenance,
        }


_PIT_DF_COLUMNS = [
    "series_key",
    "source",
    "source_series_id",
    "asof_date",
    "vintage_date",
    "obs_date",
    "value",
    "value_raw",
    "frequency",
    "units",
    "seasonal_adjustment",
    "realtime_start",
    "realtime_end",
    "ingested_at",
    "provenance",
]


def create_pit_dataframe(observations: list[PITObservation]) -> pd.DataFrame:
    """Create a canonical PIT DataFrame from observations.

    Schema columns: series_key, source, source_series_id, asof_date,
    vintage_date, obs_date, value, value_raw, frequency, units,
    seasonal_adjustment, realtime_start, realtime_end, ingested_at, provenance.
    """
    if not observations:
        return pd.DataFrame(columns=_PIT_DF_COLUMNS)

    data = [obs.to_dict() for obs in observations]
    df = pd.DataFrame(data)

    date_cols = ["asof_date", "vintage_date", "obs_date", "realtime_start", "realtime_end"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    if "ingested_at" in df.columns:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"])

    return df


def create_wide_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a wide view for modeling from PIT DataFrame.

    Args:
        df: PIT DataFrame with single asof_date.

    Returns:
        Wide DataFrame with index=obs_date, columns=series_key, values=value.
    """
    if df.empty:
        return pd.DataFrame()

    asof_dates = df["asof_date"].unique()
    if len(asof_dates) > 1:
        raise ValueError(
            f"Wide view requires single asof_date, got {len(asof_dates)} different dates"
        )

    return df.pivot(index="obs_date", columns="series_key", values="value")
