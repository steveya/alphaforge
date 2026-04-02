# Data Sources Guide

Alphaforge data access is mediated by `DataContext`, which wires source names to source objects.

## Unified Data Layer

The unified data layer provides a single `SourceAdapter` protocol for all data sources, whether they serve PIT macro data, market OHLCV, or bulk positioning data. Key components:

- **`SourceAdapter`** — protocol that every adapter implements (`fetch`, `prefetch`, `list_entities`)
- **`SourceAdapterBase`** — mixin with default `fetch_many` (iterates) and `prefetch` (no-op)
- **`CacheLayer`** — DuckDB-backed cache with separate PIT and market observation tables
- **`FetchResult`** — unified return type with `data`, `source`, `dataset`, `is_pit`, `cached_at`

### Registering adapters

```python
from alphaforge.data.context import DataContext
from alphaforge.data.sources.tiingo import TiingoAdapter
from alphaforge.data.sources.fred import FREDSourceAdapter

ctx = DataContext(
    sources={"tiingo": legacy_source},    # legacy path (backward compat)
    calendars={"XNYS": cal},
    store=store,
    adapters={                             # unified path
        "tiingo": TiingoAdapter(api_key="..."),
        "fred": FREDSourceAdapter(api_key="..."),
    },
    default_sources={"market.ohlcv": "tiingo", "macro.fred": "fred"},
)
```

### Fetching data

```python
from alphaforge.data.query import Query

# Unified fetch — routes to the correct adapter automatically
result = ctx.fetch(Query(table="market.ohlcv", entities=["SPY"], start=start, end=end))
result.data   # DataFrame
result.source # "tiingo"

# Legacy path still works
panel = ctx.fetch_panel("tiingo", query)
```

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

## Practical recommendation

For production pipelines, keep source instantiation and registry/version pins explicit in one bootstrap module so dataset builds remain reproducible over time.

## Local futures artifacts

The First Rate futures integration is configured through explicit paths, YAML config,
or environment variables. The loader ingests a flat raw directory of
`*_5min.txt` contract files, writes canonical parquet artifacts under a separate
artifact root, and exposes those artifacts through a `SourceAdapter`.

See [First Rate futures guide](first-rate-futures.md) for the expected folder
structure, environment variables, and dataset names.
