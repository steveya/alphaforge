# Public Web API

`alphaforge.data.public_web` contains source loaders for public datasets that do
not justify a dedicated adapter package but still need stable schemas,
deterministic normalization, and point-in-time-safe `asof_utc` handling.

These public-web loaders are part of the legacy/raw-loader `DataSource`
surface. They remain supported because the public-web pack has not been fully
adapterized, but they are not the canonical general loading contract for new
code outside this loader family.

## Public entry points

- `alphaforge.data.public_web.default_public_web_sources()` builds the default
  raw-loader registry used by compatibility-oriented `DataContext` bootstrap
  code.
- `alphaforge.data.public_web` re-exports the concrete public loader classes.
- `alphaforge.default_public_web_sources()` mirrors the same registry at the
  package root for convenience.

## Default source catalog

### Registry-backed APIs

| Source name | Class | Table(s) |
| --- | --- | --- |
| `bcb_sgs` | `BCBSGSDataSource` | `bcb_sgs_series` |
| `bea` | `BEADataSource` | `bea_series` |
| `bls` | `BLSDataSource` | `bls_series` |
| `destatis_genesis` | `DestatisGenesisDataSource` | `destatis_series` |
| `ecb_sdmx` | `ECBSDMXDataSource` | `ecb_sdmx_series` |
| `eia` | `EIADataSource` | `eia_series` |
| `eurostat` | `EurostatDataSource` | `eurostat_series` |
| `ibge_sidra` | `IBGESidraDataSource` | `ibge_sidra_series` |

### Tabular and document loaders

| Source name | Class | Table(s) |
| --- | --- | --- |
| `cme_productslate` | `CMEProductSlateSource` | `cme.productslate.reference` |
| `ec_weekly_oil_bulletin` | `ECWeeklyOilBulletinDataSource` | `ec_oil_bulletin_weekly` |
| `eurex_refdata_contracts` | `EurexRefdataContractsSource` | `eurex.refdata.contracts` |
| `eurex_stats_daily` | `EurexStatsDailySource` | `eurex.stats.daily` |
| `ezoic_adrevenue_daily` | `EzoicAdRevenueDailySource` | `ezoic.adrevenue.daily` |
| `frb_term_structure` | `FRBTermStructureBenchmarkSource` | `frb.term_structure` |
| `lch_cdsclear_daily` | `LCHCDSClearDailySource` | `lch.cdsclear.daily` |

### Archive and batch loaders

| Source name | Class | Table(s) |
| --- | --- | --- |
| `b3_historical_quotes` | `B3HistoricalQuotesDataSource` | `b3_equity_quotes_daily` |
| `cftc_cot` | `CFTCCoTSource` | `cftc.cot.tff` |
| `cftc_cot_disagg` | `CFTCDisaggregatedCoTSource` | `cftc.cot.disagg` |
| `cftc_swaps_weekly` | `CFTCWeeklySwapsSource` | `cftc.swaps.weekly` |

Archive-backed CFTC loaders now raise an explicit failure when a requested
archive cannot be downloaded or parsed, rather than silently skipping the bad
year and returning partial history.

### Outliers and provider-specific loaders

| Source name | Class | Table(s) |
| --- | --- | --- |
| `anp_fuel_prices` | `ANPFuelPricesDataSource` | `anp_fuel_prices_weekly` |
| `dtcc_ppd` | `DTCCPPDSource` | `dtcc.ppd.events`, `dtcc.ppd.daily` |
| `mof_jgb_yields` | `MOFJGBYieldCurveSource` | `mof.jgb.yields` |
| `philadelphia_spf` | `PhiladelphiaSPFMeanLevelSource` | `philadelphia.spf.mean_level` |

`DTCCPPDSource` remains the low-level raw-loader surface for DTCC. Canonical
adapter-backed access for the first DTCC families now lives in
`alphaforge.data.sources.dtcc` through `DTCCFXAdapter` (`dtcc.fx`) and
`DTCCIRSAdapter` (`dtcc.irs`).

## Shared helper layers

These helpers reduce boilerplate inside `alphaforge.data.public_web`, but they
are still internal implementation utilities rather than a broad compatibility
promise for third-party extensions.

| Helper module | Purpose |
| --- | --- |
| `base.py` | shared HTTP client setup, empty-frame construction, and finalization |
| `finalize.py` | schema-aware projection, sorting, entity/date filtering, and `asof_utc` handling |
| `schema_helpers.py` | concise `TableSchema` builders for daily panels, single-value series, and event tables |
| `registry_api.py` | shallow base for registry-driven entity loaders |
| `tabular.py` | helpers for HTML/XLSX/CSV document parsing and column resolution |
| `archive.py` | helpers for archive discovery, deterministic fetch planning, yearly archive planning, and ZIP member reads |

## Authoring guidance

For new loaders or refactors, use the contributor workflow in the
[public-web source authoring guide](../guides/public-web-source-authoring.md).
That guide covers helper-family selection, registry and export wiring,
defensive parsing, and the required test and doc updates.
