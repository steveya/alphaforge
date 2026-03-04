# Alphaforge Public-Web Loaders: Common Conventions

This spec defines **free / publicly downloadable** data loaders that integrate with the `alphaforge.data.DataSource` interface.

## Alphaforge interface (must match repo)

A loader is a `DataSource` with:

- `name: str`
- `schemas() -> dict[str, TableSchema]`
- `fetch(q: Query) -> pd.DataFrame`

Where `Query` provides:

- `q.table: str`
- `q.columns: Sequence[str]`
- optional filters: `q.start`, `q.end` (tz-aware UTC), `q.entities`
- optional: `q.asof` (tz-aware UTC)

`fetch()` MUST:
- return a **long** DataFrame with the table schema’s `time_column` + `entity_column` + requested columns
- apply pushdowns when possible: **columns**, **date range**, **entities**, and **asof** (when dataset supports it)
- return UTC timestamps for any timestamp columns; `Query` already normalizes `start/end/asof` to UTC.

## Project layout (create new package)

Create:

- `alphaforge/data/public_web/`
  - `http.py` (download/cache/retry)
  - `parsing.py` (zip/csv/xlsx/html parsing helpers)
  - `utils.py` (normalization + entity-id helpers + tenor bucketing)
  - one module per datasource (see dataset specs)
  - `__init__.py` exporting sources
  - optionally: `registry.py` (map `source.name -> instance`)

## Storage & caching

All loaders should implement **download caching** to avoid re-downloading the same artifact.

Minimum requirement:
- raw download cache: `~/.cache/alphaforge/public_web/<source>/<yyyy>/<mm>/<artifact_name>`
- cache key uses URL + query params + (optional) content hash

Recommended:
- respect `ETag` / `Last-Modified` headers when present
- backoff/retry on 429/5xx
- set a consistent `User-Agent` string, e.g. `alphaforge/<version> (+https://github.com/...)`

## Standard columns

Unless otherwise specified, tables should include:

- `date` (tz-aware UTC at midnight for daily series)
- `entity_id` (string)
- `asof_utc` (tz-aware UTC). For “event-level” tables, `asof_utc` equals `reported_at_utc` if present, else ingestion timestamp.
- numeric columns should be floats (or ints when exact).

## Entity-id conventions

Use structured, parseable entity IDs:

- use **dot-separated namespaces**: `rates.irs.usd.2y`, `rates.irs.eur.5y`, `fx.spot.eurusd`, etc.
- include the source if ambiguous: `dtccppd.rates.irs.usd.2y`
- keep entity IDs stable over time.

Provide helper:

`make_entity_id(parts: list[str]) -> str` that lowercases and joins with dots.

## Tenor bucketing (shared)

For event-level derivatives data, standardize tenors into buckets to support aggregation tables:

- `1m, 3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 15y, 20y, 30y, 40y, 50y`
- a function: `bucket_tenor(raw: str|float|pd.Timedelta) -> str|None`

## Testing policy

All loaders must have:
- **unit tests** for parsers (using fixtures under `tests/fixtures/public_web/`)
- **contract tests** per table:
  - required columns present
  - time column parseable to tz-aware UTC
  - entity column string
  - `asof_utc` present
- no network in CI by default (live tests optional behind env var)

## Tables to implement (names chosen here)

- `dtcc.ppd.events`
- `dtcc.ppd.daily`
- `cftc.swaps.weekly`
- `eurex.stats.daily`
- `eurex.refdata.contracts`
- `lch.cdsclear.daily`
- `cme.productslate.reference`
- `ezoic.adrevenue.daily`
