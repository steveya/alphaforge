# Data Sources Guide

Alphaforge data access is mediated by `DataContext`.

For new code, the canonical public loading path is:

- build a context with `DataContext.from_adapters(...)`
- call `DataContext.load(...)`, `fetch_many(...)`, or `prefetch(...)`

The older `DataContext.sources` and `fetch_panel(...)` path remains available
only for backward compatibility and raw-loader flows that have not migrated to
adapters yet.

## Unified Data Layer

The unified data layer provides a single `SourceAdapter` protocol for all data
sources, whether they serve PIT macro data, market OHLCV, or bulk positioning
data. Key components:

- **`SourceAdapter`** — protocol that every adapter implements (`fetch`, `prefetch`, `list_entities`)
- **`SourceAdapterBase`** — mixin with default `fetch_many` (iterates) and `prefetch` (no-op)
- **`CacheLayer`** — DuckDB-backed cache with separate PIT and market observation tables
- **`FetchResult`** — unified return type with `data`, `source`, `dataset`, `is_pit`, `cached_at`

### Registering adapters

```python
from alphaforge.data.context import DataContext
from alphaforge.data.sources.tiingo import TiingoAdapter
from alphaforge.data.sources.fred import FREDSourceAdapter

ctx = DataContext.from_adapters(
    TiingoAdapter(api_key="..."),
    FREDSourceAdapter(api_key="..."),
    calendars={"XNYS": cal},
    store=store,
    default_sources={"market.ohlcv": "tiingo", "macro.fred": "fred"},
)
```

`from_adapters(...)` derives the adapter map for you and automatically sets
default sources for datasets that are served by exactly one adapter.

If a dataset is served by more than one adapter, canonical routing now requires
either:

- a `default_sources` entry for that dataset, or
- an explicit `source=` on the fetch/load call

Alphaforge no longer guesses by taking the first registered adapter for an
ambiguous dataset.

### Fetching data

```python
# Happy-path single-table load
result = ctx.load(
    "market.ohlcv",
    columns=["close", "volume"],
    entities=["SPY"],
    start=start,
    end=end,
)
result.data   # DataFrame
result.source # "tiingo"

# Canonical batch fetch preserves input order and lets adapters optimize
results = ctx.fetch_many(
    [
        Query(table="market.ohlcv", entities=["SPY"], start=start, end=end),
        Query(table="macro.fred", entities=["GDP"], start=start, end=end),
    ]
)
```

`fetch_many(...)` groups queries by the resolved adapter and delegates through
the adapter batch contract, so cache-aware sources can optimize multi-query
loads without changing the caller surface.

### Compatibility boundary

These older patterns still work during migration, but they are not the
preferred public API for new code:

- `ctx.sources[...]`
- `ctx.fetch_panel(...)`
- direct `DataSource` usage as the primary loading contract

Keep using them only where a loader family has not been migrated to
`SourceAdapter` yet, or where a raw `PanelFrame` conversion is still required
internally.

### Entry-point discovery

Adapters are registered as `alphaforge.source_adapters` entry points. Third-party packages can add their own adapters by declaring an entry point in their `pyproject.toml`:

```toml
[project.entry-points."alphaforge.source_adapters"]
my_source = "my_package.adapters:MyAdapter"
```

Discover all installed adapters:

```python
from alphaforge.data.sources import discover_adapters

available = discover_adapters()
# {'tiingo': <class TiingoAdapter>, 'fred': <class FREDSourceAdapter>, ...}
```

## Source categories

- Local/in-memory sources for tests and prototyping
- Local configurable futures sources such as `alphaforge.futures.FirstRateFuturesLoader`
- Public web source pack under `alphaforge.data.public_web`
- FRED-style macro sources
- Unified adapters under `alphaforge.data.sources`

## Query contract

Most source fetches are driven by `alphaforge.data.query.Query`, including:

- `table`
- `columns`
- `start` / `end`
- `entities`
- `asof`
- `grid`

## Registries

Some public web sources are configured through YAML registries in `alphaforge/data/registries`.

For the public-web loader pack specifically, see the
[public-web source authoring guide](public-web-source-authoring.md) for
helper-family selection, registry wiring, and validation expectations.

## Practical recommendation

For production pipelines, keep adapter instantiation and registry/version pins
explicit in one bootstrap module so dataset builds remain reproducible over
time.

## Operational helpers

For source monitoring and recurring archive-backed ingestion, see
[Source Operations](source-operations.md).

## Local futures artifacts

The First Rate futures integration is configured through explicit paths, YAML config,
or environment variables. The loader ingests a flat raw directory of
`*_5min.txt` contract files, writes canonical parquet artifacts under a separate
artifact root, and exposes those artifacts through a `SourceAdapter`.

See [First Rate futures guide](first-rate-futures.md) for the expected folder
structure, environment variables, and dataset names.
