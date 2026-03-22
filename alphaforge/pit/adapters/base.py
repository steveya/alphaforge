"""Base adapter interface for PIT data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

from alphaforge.pit.observation import PITObservation, SeriesMetadata

if TYPE_CHECKING:
    from alphaforge.time.ref_period import RefFreq, RefPeriod


class PITAdapter(ABC):
    """Base class for point-in-time data adapters.

    Concrete subclasses fetch vintage data from external sources (FRED/ALFRED,
    ECB, BoE, etc.) and return :class:`PITObservation` records.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Adapter name/identifier."""

    @abstractmethod
    def supports_pit(self, series_id: str) -> bool:
        """Check if a series supports point-in-time retrieval.

        Args:
            series_id: Source-specific series identifier.

        Returns:
            True if PIT is supported, False otherwise.
        """

    @abstractmethod
    def list_vintages(self, series_id: str) -> list[date]:
        """List available vintage dates for a series.

        Args:
            series_id: Source-specific series identifier.

        Returns:
            Sorted list of vintage dates.

        Raises:
            PITNotSupportedError: If series doesn't support vintages.
            SourceFetchError: If fetching fails.
        """

    @abstractmethod
    def fetch_asof(
        self,
        series_id: str,
        asof_date: date,
        start: date | None = None,
        end: date | None = None,
        *,
        metadata: SeriesMetadata | None = None,
        **kwargs: Any,
    ) -> list[PITObservation]:
        """Fetch observations as they were known on *asof_date*.

        Args:
            series_id: Source-specific series identifier.
            asof_date: Point-in-time evaluation date.
            start: Optional start date for observation period.
            end: Optional end date for observation period.
            metadata: Optional series metadata.

        Returns:
            List of PIT observations.

        Raises:
            PITNotSupportedError: If series doesn't support PIT.
            VintageNotFoundError: If no vintage available at asof_date.
            SourceFetchError: If fetching fails.
        """

    def fetch_asof_ref(
        self,
        series_id: str,
        asof_date: date,
        start_ref: str | RefPeriod | None = None,
        end_ref: str | RefPeriod | None = None,
        *,
        freq: RefFreq | None = None,
        metadata: SeriesMetadata | None = None,
    ) -> list[PITObservation]:
        """Optional ref-period snapshot query."""
        raise NotImplementedError("Ref-period snapshot queries not supported.")

    def fetch_revisions_ref(
        self,
        series_id: str,
        ref: str | RefPeriod,
        start_asof: date | None = None,
        end_asof: date | None = None,
        *,
        freq: RefFreq | None = None,
        metadata: SeriesMetadata | None = None,
    ) -> pd.Series:
        """Optional ref-period revision timeline query."""
        raise NotImplementedError("Ref-period revision queries not supported.")

    def fetch_vintage(
        self,
        series_id: str,
        vintage_date: date,
        start: date | None = None,
        end: date | None = None,
    ) -> list[PITObservation]:
        """Fetch a specific vintage of observations.

        Default implementation delegates to :meth:`fetch_asof`.
        """
        return self.fetch_asof(series_id, vintage_date, start, end)

    def list_pit_observations_asof(
        self,
        *,
        series_key: str,
        obs_date: date,
        asof_date: date,
    ) -> pd.DataFrame:
        """List all PIT observations for a series/obs_date up to an as-of date.

        Returns:
            DataFrame with columns: series_key, obs_date, asof_utc, value.
        """
        raise NotImplementedError("PIT observation listing not supported.")

    def list_pit_observations_asof_multi(self, requests: pd.DataFrame) -> pd.DataFrame:
        """Optional batched PIT observation listing.

        Expected request columns: request_id, series_key, obs_date, asof_date.
        """
        raise NotImplementedError("Batched PIT observation listing not supported.")
