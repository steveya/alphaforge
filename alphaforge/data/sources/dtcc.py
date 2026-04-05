"""DTCC adapters over the low-level DTCC PPD raw loader.

`DTCCPPDSource` remains the provider-specific raw-loader implementation.
The adapter layer owns canonical `SourceAdapter` routing, cache/prefetch
behavior, and PIT transform wiring on top of that raw loader.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, cast

import duckdb
import pandas as pd

from ..adapter import SourceAdapterBase
from ..cache_layer import CacheLayer
from ..public_web.dtcc_ppd import DTCCPPDSource
from ..query import Query
from ..source import DataSource
from ..transforms.dtcc_pit import dtcc_daily_to_pit_observations
from ..types import CacheManifest, FetchResult

_DEFAULT_RAW_COLUMNS = (
    "trade_count",
    "notional_sum",
    "price_mean",
    "price_std",
    "notional_median",
    "trade_count_large",
    "dv01_proxy_sum",
)


def _build_dtcc_source(
    *,
    source: DataSource | None,
    source_kwargs: dict[str, Any],
    default_asset_codes: tuple[str, ...] | None = None,
) -> DataSource:
    if source is not None and source_kwargs:
        raise ValueError(
            "Pass either a preconfigured source or DTCCPPDSource keyword "
            "arguments, not both."
        )

    if source is not None:
        return source

    resolved_kwargs = dict(source_kwargs)
    if default_asset_codes is not None:
        resolved_kwargs.setdefault("asset_codes", default_asset_codes)
    source_factory = cast(Any, DTCCPPDSource)
    return source_factory(**resolved_kwargs)


class DTCCPPDAdapterBase(SourceAdapterBase):
    """Shared cache-aware adapter plumbing over `DTCCPPDSource`.

    Subclasses define the adapter-facing dataset contract and the PIT
    transform, while this base owns raw-loader fetch construction and cache
    lifecycle management.
    """

    source_name: str
    datasets: frozenset[str]
    raw_table = DTCCPPDSource.DAILY_TABLE
    raw_columns = _DEFAULT_RAW_COLUMNS

    def __init__(
        self,
        *,
        source: DataSource,
        cache_conn: Optional[duckdb.DuckDBPyConnection] = None,
    ) -> None:
        self._source = source
        self._cache: CacheLayer | None = None
        if cache_conn is not None:
            self._cache = CacheLayer(cache_conn)

    def _require_dataset(self, dataset: str) -> None:
        if dataset not in self.datasets:
            raise KeyError(
                f"{type(self).__name__} does not serve dataset '{dataset}'. "
                f"Available: {sorted(self.datasets)}"
            )

    def _raw_table_for_dataset(self, dataset: str) -> str:
        self._require_dataset(dataset)
        return self.raw_table

    def _raw_columns_for_dataset(self, dataset: str) -> tuple[str, ...]:
        self._require_dataset(dataset)
        return self.raw_columns

    def _raw_fetch(
        self,
        dataset: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        return self._source.fetch(
            Query(
                table=self._raw_table_for_dataset(dataset),
                columns=list(self._raw_columns_for_dataset(dataset)),
                start=start,
                end=end,
            )
        )

    def _to_pit(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fetch(
        self,
        query: Query,
        *,
        max_staleness: Optional[timedelta] = None,
    ) -> FetchResult:
        """Fetch DTCC data through the canonical adapter contract."""
        dataset = query.table
        self._require_dataset(dataset)
        entities = list(query.entities or [])

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
                    dataset=dataset,
                    is_pit=True,
                    cached_at=cached_at,
                )

        raw_df = self._raw_fetch(dataset, query.start, query.end)
        pit_df = self._to_pit(dataset, raw_df)

        if self._cache is not None and not pit_df.empty:
            self._cache.store(
                pit_df,
                dataset=dataset,
                source=self.source_name,
                is_pit=True,
            )

        if entities and not pit_df.empty:
            pit_df = pit_df[pit_df["series_key"].isin(entities)]

        return FetchResult(
            data=pit_df,
            source=self.source_name,
            dataset=dataset,
            is_pit=True,
            cached_at=None,
        )

    def prefetch(
        self,
        dataset: str,
        asof_range: tuple[date, date] | None = None,
    ) -> CacheManifest:
        """Bulk fetch and cache all rows for an adapter dataset."""
        self._require_dataset(dataset)
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
        """List cached entity keys for a DTCC adapter dataset."""
        self._require_dataset(dataset)
        if self._cache is not None:
            manifest = self._cache.get_manifest(dataset=dataset, source=self.source_name)
            if manifest is not None:
                return manifest.entity_keys
        return []


class DTCCAdapter(DTCCPPDAdapterBase):
    """Built-in canonical adapter for the generic DTCC PPD dataset.

    Callers no longer need to inject a raw fetch function. By default this
    adapter constructs and owns a `DTCCPPDSource` instance internally.
    Tests and advanced callers may still inject a preconfigured raw source.
    """

    source_name = "dtcc"
    datasets = frozenset({"dtcc.ppd"})

    def __init__(
        self,
        *,
        source: DataSource | None = None,
        cache_conn: Optional[duckdb.DuckDBPyConnection] = None,
        **source_kwargs: Any,
    ) -> None:
        resolved_source = _build_dtcc_source(
            source=source,
            source_kwargs=source_kwargs,
        )
        super().__init__(source=resolved_source, cache_conn=cache_conn)

    def _to_pit(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        self._require_dataset(dataset)
        return dtcc_daily_to_pit_observations(raw_df)


class _FilteredDTCCPPDAdapter(DTCCPPDAdapterBase):
    """Shared base for concrete DTCC product-family adapters."""

    key_prefix: str
    pit_source_name: str
    entity_prefix: str
    default_asset_codes: tuple[str, ...] | None = None

    def __init__(
        self,
        *,
        source: DataSource | None = None,
        cache_conn: Optional[duckdb.DuckDBPyConnection] = None,
        **source_kwargs: Any,
    ) -> None:
        resolved_source = _build_dtcc_source(
            source=source,
            source_kwargs=source_kwargs,
            default_asset_codes=self.default_asset_codes,
        )
        super().__init__(source=resolved_source, cache_conn=cache_conn)

    def _filter_raw_df(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty or "entity_id" not in raw_df.columns:
            return raw_df
        entity_ids = raw_df["entity_id"].astype(str)
        return raw_df[entity_ids.str.startswith(self.entity_prefix)].copy()

    def _to_pit(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
        self._require_dataset(dataset)
        filtered = self._filter_raw_df(raw_df)
        return dtcc_daily_to_pit_observations(
            filtered,
            key_prefix=self.key_prefix,
            source_name=self.pit_source_name,
        )


class DTCCFXAdapter(_FilteredDTCCPPDAdapter):
    """Canonical DTCC adapter for FX forwards and swaps."""

    source_name = "dtcc_fx"
    datasets = frozenset({"dtcc.fx"})
    key_prefix = "dtcc.fx."
    pit_source_name = "dtcc_ppd_fx"
    entity_prefix = "dtccppd.fx."
    default_asset_codes = ("FX",)


class DTCCIRSAdapter(_FilteredDTCCPPDAdapter):
    """Canonical DTCC adapter for interest rate swaps."""

    source_name = "dtcc_irs"
    datasets = frozenset({"dtcc.irs"})
    key_prefix = "dtcc.irs."
    pit_source_name = "dtcc_ppd_irs"
    entity_prefix = "dtccppd.rates.interest_rate_swap."
    default_asset_codes = ("IR",)
