# Spec 11 — Conventions (Inflation Nowcast Public Web Loaders)

## Context / placement
All public web data loaders live in: `alphaforge/data/public_web/`.

Assume the repo already contains shared helpers created previously (do **not** re-implement unless missing):
- `http.py` (GET JSON/bytes, retries/backoff, user-agent, optional cache)
- `parsing.py` (date coercion, safe_float, ensure_utc, etc.)
- `registry.py` (load YAML registries, validate keys)
- other small utils (chunking, caching helpers)

If any helper is missing, implement the minimal subset required to satisfy these specs (prefer reuse).

## Goal
Implement **public (non-commercial)** data sources commonly used in central-bank-style inflation nowcasting, suitable for building data loaders in AlphaForge.

Covered sources (public):
- US: BLS CPI, BEA PCE price indices, EIA energy prices
- EU/EA: Eurostat HICP (preferred), ECB SDMX HICP (optional alternative), EC Weekly Oil Bulletin (fuel prices)
- DE: Destatis GENESIS CPI/HICP (official national series)
- BR: IBGE SIDRA, BCB SGS, ANP fuel prices
- BR optional macro-financial: B3 historical quotes (file-based, public)

Commercial sources are explicitly excluded: Bloomberg, GfK scanner, Amadeus, etc.

## AlphaForge integration contract (assumed)
Implement each provider as a `DataSource` (or your existing equivalent base class). Each DataSource should expose at least:
- `name: str`
- `schemas() -> dict[str, TableSchema]` (or your catalog schema mechanism)
- `fetch(q: Query) -> pd.DataFrame` returning **long** data

### Standard output columns
Minimum output for time series tables:
- `date`: tz-aware UTC Timestamp (observation period end unless noted)
- `entity_id`: stable string identifier for the series (provider series id or synthetic id)
- `value`: float
- `asof_utc`: tz-aware UTC Timestamp indicating retrieval/as-of time

Optional but okay:
- units, frequency, provider metadata columns (keep lean)

### Date semantics
- Monthly macro series: represent `date` as the **month end** UTC timestamp.
- Quarterly series: represent `date` as the **quarter end** UTC timestamp.
- Daily series: `date` is the day timestamp (UTC midnight is fine).
- Weekly series: choose a consistent convention:
  - if provider gives a specific date, use it; otherwise treat as “week ending” date (UTC midnight). Document it per source.

### PIT / vintages
Most of these providers do **not** supply real-time vintage revisions through the API. Therefore:
- set `asof_utc = q.asof` if present, else `now_utc()` at fetch time.
- revisions won’t be reconstructable unless the provider supports it; do not simulate vintages.

### Registries (required pattern)
For APIs where “series definition” is a parameter bundle (e.g., BEA, EIA v2 routes, Eurostat dataset/dim keys, ECB SDMX flow+key, Destatis codes, SIDRA table variables):
- Create a YAML registry in `alphaforge/data/registries/`.
- The DataSource loads registry entries; `q.entities` must refer to registry keys (stable `entity_id`).

Registry files introduced by these specs:
- `bea_series.yaml`
- `eia_series.yaml`
- `eurostat_series.yaml`
- `ecb_sdmx_series.yaml`
- `destatis_series.yaml`
- `ibge_sidra_series.yaml`

## Naming conventions
### DataSource class naming
Place file and class under `alphaforge/data/public_web/`:
- `bls.py` → `BLSDataSource`
- `bea.py` → `BEADataSource`
- `eia.py` → `EIADataSource`
- `eurostat.py` → `EurostatDataSource`
- `ecb_sdmx.py` → `ECBSDMXDataSource`
- `destatis_genesis.py` → `DestatisGenesisDataSource`
- `ec_weekly_oil_bulletin.py` → `ECWeeklyOilBulletinDataSource`
- `ibge_sidra.py` → `IBGESidraDataSource`
- `bcb_sgs.py` → `BCBSGSDataSource`
- `anp_fuel_prices.py` → `ANPFuelPricesDataSource`
- `b3_historical_quotes.py` → `B3HistoricalQuotesDataSource` (optional)

### Table names
Use one primary table per provider unless explicitly noted:
- `bls_series`
- `bea_series`
- `eia_series`
- `eurostat_series`
- `ecb_sdmx_series`
- `destatis_series`
- `ec_oil_bulletin_weekly`
- `ibge_sidra_series`
- `bcb_sgs_series`
- `anp_fuel_prices_weekly`
- `b3_equity_quotes_daily`

## Tests (required)
For each provider:
- A unit test that calls `fetch()` for a small range and a known entity.
- Assert required columns exist, dtypes are sensible, and date filtering works.
- Tests must be resilient (avoid “latest week only” assertions). Prefer fixed historical window.
- Mark tests as “network” if your repo uses such markers; otherwise keep them minimal.

## Rate limiting / robustness
- Respect provider rate limits; implement chunking for batched endpoints (BLS, etc.).
- Ensure retries for transient failures are in shared http helper (reuse existing).
- Cache large static files where appropriate (ANP, B3) using existing cache utilities.