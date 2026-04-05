# DataContext

`DataContext.fetch(...)`, `fetch_many(...)`, and `prefetch(...)` are the
canonical public data-loading entry points for new code.

For the shortest happy path, build contexts with `DataContext.from_adapters(...)`
and load a table with `DataContext.load(...)`.

If multiple adapters serve the same dataset, canonical loads must disambiguate
through `default_sources` or an explicit `source=` override. Alphaforge no
longer falls back to an arbitrary first-registered adapter for shared datasets.

`DataContext.sources` and `fetch_panel(...)` remain available as
backward-compatibility surfaces for legacy `DataSource`-backed loaders, but
they are no longer the preferred external loading contract.

::: alphaforge.data.context
