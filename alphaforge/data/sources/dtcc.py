"""DTCCAdapter — Bulk SourceAdapter for DTCC PPD data.

Fetches from DTCC (via DTCCPPDSource), transforms with
dtcc_daily_to_pit_observations, and caches the full bulk result.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

import duckdb
import pandas as pd

from ..adapter import SourceAdapterBase
from ..cache_layer import CacheLayer
from ..query import Query
from ..transforms.dtcc_pit import dtcc_daily_to_pit_observations
from ..types import CacheManifest, FetchResult

logger = logging.getLogger(__name__)


class DTCCAdapter(SourceAdapterBase):
    """Cache-aware bulk adapter for DTCC PPD data.

    Parameters
    ----------
    raw_fetcher : callable(start, end) -> DataFrame
        Function that fetches raw DTCC data.
    cache_conn : duckdb.DuckDBPyConnection | None
        DuckDB connection for caching.
    """

    source_name = "dtcc"
    datasets = frozenset({"dtcc.ppd"})

    def __init__(
        self,
        raw_fetcher: Callable[[Optional[pd.Timestamp], Optional[pd.Timestamp]], pd.DataFrame],
        cache_conn: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> None:
        self._raw_fetcher = raw_fetcher
        self._cache: CacheLayer | None = None
        if cache_conn is not None:
            self._cache = CacheLayer(cache_conn)

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: Optional[timedelta] = None,
    ) -> FetchResult:
        """Fetch DTCC data. Bulk fetch on miss, serve from cache on hit."""
        entities = list(query.entities or [])

        # Try cache
        if entities and self._cache is not None:
            cached_frames = []
            all_cached = True
            cached_at = None

            for series_key in entities:
                result = self._cache.lookup(
                    series_key=series_key,
                    dataset="dtcc.ppd",
                    source=self.source_name,
                    is_pit=True,
                    max_staleness=max_staleness,
                )
                if result is not None:
                    df, cached_at = result
                    cached_frames.append(df)
                else:
                    all_cached = False
                    break

            if all_cached and cached_frames:
                combined = pd.concat(cached_frames, ignore_index=True)
                return FetchResult(
                    data=combined,
                    source=self.source_name,
                    dataset=query.table,
                    is_pit=True,
                    cached_at=cached_at,
                )

        # Bulk fetch + transform
        raw_df = self._raw_fetcher(query.start, query.end)
        pit_df = dtcc_daily_to_pit_observations(raw_df)

        # Cache
        if self._cache is not None and not pit_df.empty:
            self._cache.store(
                pit_df,
                dataset="dtcc.ppd",
                source=self.source_name,
                is_pit=True,
            )

        # Filter
        if entities and not pit_df.empty:
            pit_df = pit_df[pit_df["series_key"].isin(entities)]

        return FetchResult(
            data=pit_df,
            source=self.source_name,
            dataset=query.table,
            is_pit=True,
            cached_at=None,
        )

    def prefetch(
        self,
        dataset: str,
        asof_range: tuple[date, date] | None = None,
    ) -> CacheManifest:
        """Bulk fetch and cache all DTCC data."""
        start = pd.Timestamp(asof_range[0]) if asof_range else None
        end = pd.Timestamp(asof_range[1]) if asof_range else None

        raw_df = self._raw_fetcher(start, end)
        pit_df = dtcc_daily_to_pit_observations(raw_df)

        if self._cache is not None and not pit_df.empty:
            self._cache.store(
                pit_df,
                dataset="dtcc.ppd",
                source=self.source_name,
                is_pit=True,
            )
            manifest = self._cache.get_manifest(dataset="dtcc.ppd", source=self.source_name)
            if manifest is not None:
                return manifest

        return CacheManifest(
            dataset=dataset,
            source=self.source_name,
            entity_keys=sorted(pit_df["series_key"].unique().tolist()) if not pit_df.empty else [],
            asof_range=asof_range or (date.min, date.min),
            populated_at=datetime.now(timezone.utc),
            row_count=len(pit_df),
        )

    def list_entities(self, dataset: str) -> list[str]:
        """List cached entity keys."""
        if self._cache is not None:
            manifest = self._cache.get_manifest(dataset=dataset, source=self.source_name)
            if manifest is not None:
                return manifest.entity_keys
        return []
