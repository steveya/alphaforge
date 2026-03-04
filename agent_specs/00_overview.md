# Public-Web Macro/Derivatives Loader Pack (Free)

Goal: implement a set of **free, publicly downloadable** datasets relevant to:

- swap market activity / repricing (OTC dissemination)
- terminal-rate repricing pressure (activity + price dispersion proxies)
- bond risk-premium regimes (listed rates futures/options volume & OI)
- FX carry / hedging waves (via proxies; this pack includes a “risk appetite” proxy series)

This pack will integrate with **Alphaforge** via `DataSource` loaders.

## Scope included in this pack

1) **DTCC PPD (CFTC) dissemination**  
   - event-level: `dtcc.ppd.events`
   - daily aggregates: `dtcc.ppd.daily`

2) **CFTC Weekly Swaps Report** (downloadable archive XLSX)  
   - `cftc.swaps.weekly`

3) **Eurex market statistics** (web tables; daily)  
   - `eurex.stats.daily`

4) **Eurex free reference data API** (contracts/mastering)  
   - `eurex.refdata.contracts`

5) **LCH CDSClear volumes** (web tables; daily)  
   - `lch.cdsclear.daily`

6) **CME ProductSlate** (direct downloadable CSV)  
   - `cme.productslate.reference`

7) **Ezoic Ad Revenue Index** (web series)  
   - `ezoic.adrevenue.daily`

## Non-goals

- paid datasets
- authenticated feeds
- tick-level exchange data
- full EMIR/UK TR data (public reports are aggregated and sometimes not cleanly automatable without ToS review)

## Acceptance criteria (done means)

For each table:
- `TableSchema` added with correct time/entity columns and required columns
- loader fetch respects `Query` filters (start/end/entities) and returns only requested columns
- returned DataFrame is long, tidy, and uses UTC timestamps
- parsing is deterministic and tested with fixtures
- example script shows how to attach sources to a `DataContext` and pull data
