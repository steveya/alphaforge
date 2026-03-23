"""FREDSourceAdapter — SourceAdapter wrapping FREDALFREDAdapter for PIT macro data.

Wraps the existing ``FREDALFREDAdapter`` (which fetches from the FRED API
with realtime_start/realtime_end) and presents it through the unified
``SourceAdapter`` interface with cache-layer integration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import duckdb
import pandas as pd

from ...pit.adapters.fred import FREDALFREDAdapter
from ..adapter import SourceAdapterBase
from ..cache_layer import CacheLayer
from ..query import Query
from ..types import FetchResult

logger = logging.getLogger(__name__)


class FREDSourceAdapter(SourceAdapterBase):
    """Cache-aware adapter for FRED/ALFRED macro data (PIT).

    Parameters
    ----------
    api_key : str
        FRED API key.
    cache_conn : duckdb.DuckDBPyConnection | None
        DuckDB connection for caching. If None, no caching.
    datasets : frozenset[str] | None
        Override the default dataset set.
    """

    source_name = "fred"
    datasets = frozenset({"macro.fred"})

    def __init__(
        self,
        api_key: str,
        cache_conn: Optional[duckdb.DuckDBPyConnection] = None,
        datasets: Optional[frozenset[str]] = None,
    ) -> None:
        self._fred = FREDALFREDAdapter(api_key=api_key)
        self._cache: CacheLayer | None = None
        if cache_conn is not None:
            self._cache = CacheLayer(cache_conn)
        if datasets is not None:
            self.__class__ = type(
                self.__class__.__name__,
                (self.__class__,),
                {"datasets": datasets},
            )

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: Optional[timedelta] = None,
    ) -> FetchResult:
        """Fetch PIT macro data for requested series."""
        entities = list(query.entities or [])
        if not entities:
            return FetchResult(
                data=pd.DataFrame(),
                source=self.source_name,
                dataset=query.table,
                is_pit=True,
                cached_at=None,
            )

        asof_date = None
        if query.asof is not None:
            asof_date = query.asof.date() if hasattr(query.asof, "date") else query.asof

        frames = []
        any_cached_at = None

        for series_id in entities:
            # Try cache
            cached = self._cache_lookup(series_id, query, max_staleness)
            if cached is not None:
                df, cached_at = cached
                frames.append(df)
                any_cached_at = cached_at
                continue

            # Fetch from FRED API via legacy adapter
            try:
                observations = self._fred.fetch_asof(
                    series_id=series_id,
                    asof_date=asof_date or datetime.now(timezone.utc).date(),
                    start=query.start.date() if query.start is not None else None,
                    end=query.end.date() if query.end is not None else None,
                )
            except Exception:
                logger.warning("FRED fetch failed for %s", series_id, exc_info=True)
                continue

            if not observations:
                continue

            # Convert PITObservation list → DataFrame
            rows = []
            for obs in observations:
                rows.append(
                    {
                        "series_key": obs.series_key,
                        "obs_date": pd.Timestamp(obs.obs_date),
                        "asof_utc": pd.Timestamp(obs.asof_date),
                        "value": obs.value,
                    }
                )
            df = pd.DataFrame(rows)
            frames.append(df)

            # Persist to cache
            self._cache_store(df, query)

        if not frames:
            combined = pd.DataFrame(columns=["series_key", "obs_date", "asof_utc", "value"])
        else:
            combined = pd.concat(frames, ignore_index=True)

        return FetchResult(
            data=combined,
            source=self.source_name,
            dataset=query.table,
            is_pit=True,
            cached_at=any_cached_at,
        )

    def list_entities(self, dataset: str) -> list[str]:
        """FRED has thousands of series — return empty by default.

        Users should know their series IDs (e.g., GDP, GDPC1, UNRATE).
        """
        return []

    # ------------------------------------------------------------------
    # Cache internals
    # ------------------------------------------------------------------

    def _cache_lookup(
        self,
        series_id: str,
        query: Query,
        max_staleness: Optional[timedelta],
    ) -> Optional[tuple[pd.DataFrame, datetime]]:
        if self._cache is None:
            return None
        return self._cache.lookup(
            series_key=series_id,
            dataset=query.table,
            source=self.source_name,
            is_pit=True,
            max_staleness=max_staleness,
        )

    def _cache_store(self, df: pd.DataFrame, query: Query) -> None:
        if self._cache is None:
            return
        self._cache.store(
            df,
            dataset=query.table,
            source=self.source_name,
            is_pit=True,
        )
