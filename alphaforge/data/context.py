from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import pandas as pd

from ..pit.accessor import PITAccessor
from ..store.duckdb_parquet import DuckDBParquetStore
from ..store.store import Store
from ..time.calendar import TradingCalendar
from .panel import PanelFrame
from .query import Query
from .source import DataSource
from .universe import EntityMetadata, Universe

if TYPE_CHECKING:
    from .adapter import SourceAdapter
    from .types import CacheManifest, FetchResult


@dataclass
class DataContext:
    """Runtime wiring for data sources, calendars, and optional PIT access.

    Parameters
    ----------
    sources : Mapping[str, DataSource]
        Legacy data source mapping kept for backward compatibility and
        raw-loader workflows.
    calendars : Mapping[str, TradingCalendar]
        Trading calendar lookup.
    store : Store
        Backing store for persistence.
    adapters : dict[str, SourceAdapter] | None
        Canonical public data-loading surface keyed by source_name
        (e.g. ``"cftc"``).
    default_sources : dict[str, str] | None
        Maps dataset → default source_name for canonical adapter routing
        (e.g. ``{"cot.tff": "cftc"}``).
    """

    sources: Mapping[str, DataSource]
    calendars: Mapping[str, TradingCalendar]
    store: Store | None
    universe: Optional[Universe] = None
    entity_meta: Optional[EntityMetadata] = None
    adapters: Optional[dict[str, "SourceAdapter"]] = field(default=None)
    default_sources: Optional[dict[str, str]] = field(default=None)
    pit: Optional[PITAccessor] = field(init=False, default=None)

    # Internal: reverse index dataset → list of source_names
    _dataset_to_sources: dict[str, list[str]] = field(
        init=False, default_factory=dict, repr=False
    )

    @classmethod
    def from_adapters(
        cls,
        *adapters: "SourceAdapter",
        calendars: Mapping[str, TradingCalendar] | None = None,
        store: Store | None = None,
        universe: Optional[Universe] = None,
        entity_meta: Optional[EntityMetadata] = None,
        default_sources: Optional[dict[str, str]] = None,
    ) -> "DataContext":
        """Build a DataContext from adapters without manual mapping boilerplate."""
        adapter_map: dict[str, "SourceAdapter"] = {}
        dataset_to_sources: dict[str, list[str]] = {}

        for adapter in adapters:
            if adapter.source_name in adapter_map:
                raise ValueError(
                    f"Duplicate adapter source_name: {adapter.source_name!r}"
                )
            adapter_map[adapter.source_name] = adapter
            for dataset in adapter.datasets:
                dataset_to_sources.setdefault(dataset, []).append(adapter.source_name)

        derived_defaults = {
            dataset: sources[0]
            for dataset, sources in dataset_to_sources.items()
            if len(sources) == 1
        }
        merged_defaults = dict(derived_defaults)
        merged_defaults.update(default_sources or {})

        return cls(
            sources={},
            calendars=dict(calendars or {}),
            store=store,
            universe=universe,
            entity_meta=entity_meta,
            adapters=adapter_map,
            default_sources=merged_defaults or None,
        )

    def __post_init__(self) -> None:
        if isinstance(self.store, DuckDBParquetStore):
            self.pit = PITAccessor(self.store.conn())

        # Build reverse index: dataset → available sources
        if self.adapters:
            for source_name, adapter in self.adapters.items():
                for ds in adapter.datasets:
                    self._dataset_to_sources.setdefault(ds, []).append(source_name)

    def fetch_panel(self, source: str, q: Query) -> PanelFrame:
        """Legacy panel-building path for DataSource-backed loaders."""
        df = self.sources[source].fetch(q)

        try:
            schema = self.sources[source].schemas().get(q.table)
        except Exception:
            schema = None

        if schema is not None:
            panel = PanelFrame.from_long(
                df,
                time_col=schema.time_column,
                entity_col=schema.entity_column,
            )
        else:
            panel = PanelFrame.from_long(df)

        # Ensure asof_utc column exists by default for PIT tracking
        if "asof_utc" not in panel.df.columns:
            if q.asof is not None:
                panel.df["asof_utc"] = q.asof
            else:
                panel.df["asof_utc"] = pd.NaT

        # If the source provides a schema and it's a daily/business series,
        # map the input date labels to session close UTC timestamps using calendar.
        if schema is not None and schema.native_freq in ("B", "D"):
            # determine calendar: prefer q.grid like 'sessions:XNYS' else default 'XNYS'
            cal_name = "XNYS"
            if q.grid and isinstance(q.grid, str) and ":" in q.grid:
                parts = q.grid.split(":", 1)
                if len(parts) == 2:
                    cal_name = parts[1]
            if cal_name in self.calendars:
                cal = self.calendars[cal_name]
                dates = pd.DatetimeIndex(panel.df.index.get_level_values("ts_utc"))
                entities = panel.df.index.get_level_values("entity_id")
                closes = [cal.session_close_utc(d) for d in dates]
                new_index = pd.MultiIndex.from_arrays(
                    [pd.DatetimeIndex(closes), entities], names=["ts_utc", "entity_id"]
                )
                panel = PanelFrame(panel.df.copy())
                panel.df.index = new_index

        # first apply start/end/entities pushdowns
        panel = panel.slice(start=q.start, end=q.end, entities=q.entities)
        # then enforce as-of filtration: drop any rows with ts_utc > q.asof
        if q.asof is not None:
            panel = panel.slice(end=q.asof)
        if self.universe is not None:
            panel = self.universe.restrict_panel(panel)
        return panel.ensure_sorted()

    # ------------------------------------------------------------------
    # Unified data layer (v2 — SourceAdapter routing)
    # ------------------------------------------------------------------

    def _resolve_source(
        self, dataset: str, source: Optional[str] = None
    ) -> "SourceAdapter":
        """Resolve the adapter for a dataset, optionally overriding source."""
        if not self.adapters:
            raise KeyError(
                f"No adapters registered. Cannot resolve dataset '{dataset}'."
            )

        if source is not None:
            if source not in self.adapters:
                raise KeyError(
                    f"Source '{source}' not registered. "
                    f"Available: {sorted(self.adapters.keys())}"
                )
            adapter = self.adapters[source]
            if dataset not in adapter.datasets:
                raise KeyError(
                    f"Source '{source}' does not serve dataset '{dataset}'. "
                    f"It serves: {sorted(adapter.datasets)}"
                )
            return adapter

        # Use default_sources mapping
        if self.default_sources and dataset in self.default_sources:
            default_src = self.default_sources[dataset]
            if default_src not in self.adapters:
                raise KeyError(
                    f"Default source '{default_src}' for dataset '{dataset}' is not "
                    f"registered. Available: {sorted(self.adapters.keys())}"
                )
            adapter = self.adapters[default_src]
            if dataset not in adapter.datasets:
                raise KeyError(
                    f"Default source '{default_src}' does not serve dataset "
                    f"'{dataset}'. It serves: {sorted(adapter.datasets)}"
                )
            return adapter

        # Fallback: find any adapter that serves this dataset
        if dataset in self._dataset_to_sources:
            source_names = self._dataset_to_sources[dataset]
            if len(source_names) > 1:
                raise KeyError(
                    f"Multiple adapters serve dataset '{dataset}': "
                    f"{sorted(source_names)}. Configure default_sources or "
                    "pass source= explicitly."
                )
            src_name = source_names[0]
            return self.adapters[src_name]

        raise KeyError(
            f"No adapter found for dataset '{dataset}'. "
            f"Known datasets: {sorted(self._dataset_to_sources.keys())}"
        )

    def fetch(
        self,
        query: Query,
        *,
        source: Optional[str] = None,
        max_staleness: Optional[timedelta] = None,
    ) -> "FetchResult":
        """Canonical fetch path: resolve an adapter and delegate."""
        adapter = self._resolve_source(query.table, source)
        return adapter.fetch(query, max_staleness=max_staleness)

    def load(
        self,
        dataset: str,
        *,
        columns: Sequence[str],
        start: Optional[pd.Timestamp | str] = None,
        end: Optional[pd.Timestamp | str] = None,
        entities: Optional[Sequence[str]] = None,
        asof: Optional[pd.Timestamp | str] = None,
        vintage: str = "latest",
        vintage_id: Optional[str] = None,
        grid: Optional[str] = None,
        source: Optional[str] = None,
        max_staleness: Optional[timedelta] = None,
    ) -> "FetchResult":
        """Happy-path source load without explicit Query construction."""
        return self.fetch(
            Query(
                table=dataset,
                columns=list(columns),
                start=start,
                end=end,
                entities=list(entities) if entities is not None else None,
                asof=asof,
                vintage=vintage,
                vintage_id=vintage_id,
                grid=grid,
            ),
            source=source,
            max_staleness=max_staleness,
        )

    def fetch_many(
        self,
        queries: list[Query],
        *,
        source: Optional[str] = None,
        max_staleness: Optional[timedelta] = None,
    ) -> list["FetchResult"]:
        """Canonical batch fetch path, grouped by resolved adapter."""
        if not queries:
            return []

        grouped_queries: dict[int, tuple["SourceAdapter", list[tuple[int, Query]]]] = {}
        for idx, query in enumerate(queries):
            adapter = self._resolve_source(query.table, source)
            adapter_key = id(adapter)
            if adapter_key not in grouped_queries:
                grouped_queries[adapter_key] = (adapter, [])
            grouped_queries[adapter_key][1].append((idx, query))

        results: list[Optional["FetchResult"]] = [None] * len(queries)
        for adapter, indexed_queries in grouped_queries.values():
            batch_queries = [query for _, query in indexed_queries]
            batch_results = adapter.fetch_many(
                batch_queries,
                max_staleness=max_staleness,
            )
            if len(batch_results) != len(batch_queries):
                raise ValueError(
                    f"Adapter '{adapter.source_name}' returned {len(batch_results)} "
                    f"results for {len(batch_queries)} queries."
                )
            for (idx, _), result in zip(indexed_queries, batch_results):
                results[idx] = result

        if any(result is None for result in results):
            raise ValueError("fetch_many() did not populate every requested result.")
        return [result for result in results if result is not None]

    def prefetch(
        self,
        dataset: str,
        *,
        source: Optional[str] = None,
        asof_range: tuple = None,
    ) -> "CacheManifest":
        """Warm cache for a dataset via the resolved adapter."""
        adapter = self._resolve_source(dataset, source)
        return adapter.prefetch(dataset, asof_range=asof_range)
