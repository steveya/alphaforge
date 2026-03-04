# Dataset Spec: DTCC PPD (CFTC) – event-level and daily aggregates

## Source overview

DTCC’s Public Price Dissemination (PPD) provides public dissemination data for CFTC-reportable swaps.  
We will store:

- **event-level prints**: `dtcc.ppd.events`
- **derived daily aggregates**: `dtcc.ppd.daily`

### Primary URL(s)

- Dashboard: https://pddata.dtcc.com/ppd/cftcdashboard
- Info center: https://pddata.dtcc.com/ppd/info-center
- Quick reference guide (format + slices): https://kgc0418-tdw-data-0.s3.amazonaws.com/gtr/static/gtr/docs/RT_PPD_quick_ref_guide.pdf

> Implementation note: the dashboard exposes downloadable **slice/cumulative** ZIP archives containing CSVs.
> The agent should implement a downloader that can fetch **daily slices** and/or a **date range**.

---

## Table 1: `dtcc.ppd.events` (event-level)

### Schema

- `time_column`: `ts_utc` (tz-aware UTC)
- `entity_column`: `entity_id`

Required columns (minimum viable):
- `ts_utc` (UTC timestamp; prefer execution time; if only dissemination time exists use that)
- `entity_id` (see entity-id rules below)
- `asof_utc` (when the event became known; use dissemination time if present, else ingestion time)
- `asset_class` (string; e.g., "Rates", "FX", ...)
- `product` (string; e.g., "IRS", "OIS", "Swaption" if available)
- `currency` (ISO code if available)
- `tenor` (normalized bucket string if determinable; else raw tenor)
- `price` (float; e.g., fixed rate / spread / premium where applicable)
- `notional` (float; note: may be capped or masked)
- `action` (e.g., NEW/AMEND/CANCEL where available)

Canonical columns:
- all required columns plus:
  - `trade_id` (if present)
  - `cleared` (bool or category if present)
  - `venue` (SEF or platform if present)
  - `effective_date`, `maturity_date` (UTC dates if present)
  - `reported_at_utc` (dissemination timestamp if present)

### Entity IDs

Construct stable entity IDs that support aggregation:

- base: `dtccppd`
- then: asset class + currency + product + tenor bucket

Example:
- `dtccppd.rates.irs.usd.2y`
- `dtccppd.rates.ois.eur.5y`

If currency or tenor missing, fall back to:
- `dtccppd.rates.irs.unknown.unknown`

### Fetch semantics

`fetch(Query)` should support:

- `q.start/q.end` filter on `ts_utc`
- `q.entities` filter on `entity_id`
- `q.asof` (optional): keep only events with `asof_utc <= q.asof`

Column pushdown:
- always include time + entity + `asof_utc`
- include only `q.columns` beyond required ones

### Data acquisition strategy

Implement two modes:

1) **Incremental** (recommended)
- For a given calendar day D, download that day’s ZIP slice.
- Parse CSV(s), normalize columns, write raw cache file.
- Cache by day; if already downloaded and unchanged, skip.

2) **Backfill**
- For a historical range, iterate by day and download slices.

### Parsing rules (robustness)

- ZIP may contain multiple CSVs. Parse all and concat.
- Normalize header names to snake_case.
- Parse timestamps with timezone; if naive, localize to UTC.
- Ensure numeric conversion for price/notional.
- Add `ingested_at_utc` (optional) and set `asof_utc` when missing.

---

## Table 2: `dtcc.ppd.daily` (derived daily aggregates)

### Purpose

Provide a **daily panel** suitable for macro backtests and regime detection, derived from `dtcc.ppd.events`.

### Schema

- `time_column`: `date` (UTC midnight)
- `entity_column`: `entity_id`

Required columns:
- `date` (UTC midnight)
- `entity_id`
- `asof_utc` (when aggregate becomes known; set to end-of-day UTC or ingestion time)
- `trade_count` (int)
- `notional_sum` (float)
- `price_mean` (float; optional if price exists)
- `price_std` (float; dispersion proxy; optional)
- `notional_median` (float; optional)
- `trade_count_large` (int; count above a notional threshold if meaningful)
- `dv01_proxy_sum` (float; OPTIONAL: approximate DV01 using tenor bucket weights)

Canonical columns:
- include all required plus:
  - `price_p10`, `price_p90` (tail behavior)
  - `notional_p90`
  - `cleared_share` (if cleared flag exists)
  - `venue_share_sef` (if venue exists)

### Aggregation logic

Daily aggregate groups:
- by `date = floor(ts_utc to date)` and `entity_id` (already includes currency/product/tenor bucket)
- compute count, sums, mean/std for price where available

`dv01_proxy_sum` (optional, but valuable):
- map tenor bucket to a rough duration scalar:
  - 2y=1.9, 5y=4.5, 10y=8.5, 30y=20 (example)
- dv01_proxy = notional * duration_scalar * 1e-4
- document clearly as *proxy*.

### Implementation requirement

Implement `dtcc.ppd.daily` as:
- either produced inside the `DTCCPPDDataSource.fetch()` when `q.table == "dtcc.ppd.daily"`
- OR as a separate “derived source” that depends on `dtcc.ppd.events` cache.

Recommended: same class supports both tables for consistency and shared cache.

---

## Tests

Fixtures:
- include a tiny sample ZIP with 1–2 CSVs (redacted) under `tests/fixtures/public_web/dtcc_ppd/`

Unit tests:
- parsing ZIP -> DataFrame shape/columns
- timestamp parsing => tz-aware UTC
- entity-id construction stability
- daily aggregation correctness (count/sum/mean)

Contract tests:
- schemas exist for both tables
- `fetch()` respects date/entity filters
