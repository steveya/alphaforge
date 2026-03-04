# Dataset Spec: Ezoic Ad Revenue Index (Daily)

## Source

URL:
- https://adrevenueindex.ezoic.com/

## Table: `ezoic.adrevenue.daily`

### Purpose

A public ad revenue index (proxy for “price of attention” / digital ad cycle).
Use as a **risk appetite / cyclical demand** state variable for:
- FX carry regimes
- cross-asset risk overlays
- (sometimes) rates risk premium conditioning

### Schema

- `time_column`: `date` (UTC midnight)
- `entity_column`: `entity_id`

Required columns:
- `date`
- `entity_id` (use `ezoic.adrevenue.index`)
- `asof_utc`
- `value` (float)

Canonical columns:
- `region` or `geo` if downloadable splits exist
- `category` if downloadable splits exist

### Acquisition

Agent should implement robust scraping:
- prefer any downloadable JSON/CSV endpoints discovered in page network calls (devtools)
- if not accessible, parse the embedded data series from HTML/JS
- cache daily

### Tests

- fixture: saved HTML page or captured JSON series
- parser test: at least N daily points, numeric values
