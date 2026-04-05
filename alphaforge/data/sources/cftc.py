"""CFTCAdapter — Bulk SourceAdapter for CFTC CoT data.

Fetches from CFTC CoT public-web sources, transforms with
``cot_to_pit_observations``, and caches the full bulk result. Subsequent
queries for individual series are served from cache.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Mapping, Optional

import duckdb
import pandas as pd

from ..adapter import SourceAdapterBase
from ..cache_layer import CacheLayer
from ..query import Query
from ..transforms.cot_pit import cot_to_pit_observations
from ..types import CacheManifest, FetchResult

logger = logging.getLogger(__name__)

_DATASET_SOURCE_NAMES = {
    "cot.tff": "cftc_cot",
    "cot.disagg": "cftc_cot_disagg",
}


class CFTCAdapter(SourceAdapterBase):
    """Cache-aware bulk adapter for CFTC Commitments of Traders data.

    Each dataset is fetched in bulk through a dataset-specific callable.
    The default backward-compatible constructor still accepts a single
    ``raw_fetcher`` for ``cot.tff``.
    """

    source_name = "cftc"
    datasets = frozenset({"cot.tff"})

    def __init__(
        self,
        raw_fetcher: Callable[[Optional[pd.Timestamp], Optional[pd.Timestamp]], pd.DataFrame]
        | None = None,
        *,
        raw_fetchers: Mapping[
            str, Callable[[Optional[pd.Timestamp], Optional[pd.Timestamp]], pd.DataFrame]
        ]
        | None = None,
        cache_conn: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> None:
        fetchers = dict(raw_fetchers or {})
        if raw_fetcher is not None:
            fetchers.setdefault("cot.tff", raw_fetcher)
        if not fetchers:
            raise ValueError("CFTCAdapter requires at least one dataset raw_fetcher")

        self._raw_fetchers = fetchers
        self._raw_fetcher = fetchers.get("cot.tff")
        self.datasets = frozenset(fetchers)
        self._cache: CacheLayer | None = None
        if cache_conn is not None:
            self._cache = CacheLayer(cache_conn)

    def _raw_fetch(
        self,
        dataset: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        if dataset == "cot.tff" and self._raw_fetcher is not None:
            return self._raw_fetcher(start, end)
        if dataset not in self._raw_fetchers:
            raise KeyError(
                f"CFTCAdapter has no raw fetcher for dataset '{dataset}'. "
                f"Available: {sorted(self._raw_fetchers)}"
            )
        return self._raw_fetchers[dataset](start, end)

    def _to_pit(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        return cot_to_pit_observations(
            raw_df,
            key_prefix=f"cftc.{dataset}.",
            source_name=_DATASET_SOURCE_NAMES.get(dataset, "cftc_cot"),
        )

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: Optional[timedelta] = None,
    ) -> FetchResult:
        """Fetch CoT data. First call triggers bulk fetch; subsequent use cache."""
        dataset = query.table
        entities = list(query.entities or [])

        # Try cache for all requested entities
        if entities and self._cache is not None:
            cached_frames = []
            all_cached = True
            cached_at = None

            for series_key in entities:
                result = self._cache.lookup(
                    series_key=series_key,
                    dataset=dataset,
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

        # Cache miss — bulk fetch + transform
        raw_df = self._raw_fetch(dataset, query.start, query.end)
        pit_df = self._to_pit(dataset, raw_df)

        # Cache the full result
        if self._cache is not None and not pit_df.empty:
            self._cache.store(
                pit_df,
                dataset=dataset,
                source=self.source_name,
                is_pit=True,
            )

        # Filter to requested entities
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
        """Bulk fetch and cache all CoT data."""
        start = pd.Timestamp(asof_range[0]) if asof_range else None
        end = pd.Timestamp(asof_range[1]) if asof_range else None

        raw_df = self._raw_fetch(dataset, start, end)
        pit_df = self._to_pit(dataset, raw_df)

        if self._cache is not None and not pit_df.empty:
            self._cache.store(
                pit_df,
                dataset=dataset,
                source=self.source_name,
                is_pit=True,
            )
            manifest = self._cache.get_manifest(dataset=dataset, source=self.source_name)
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
        """List cached entity keys, or empty if no cache."""
        if self._cache is not None:
            manifest = self._cache.get_manifest(dataset=dataset, source=self.source_name)
            if manifest is not None:
                return manifest.entity_keys
        return []
