# Dataset Spec: CFTC Weekly Swaps Report (Archive)

## Source

Archive page provides weekly files (XLS/XLSX) by date.

URL:
- https://www.cftc.gov/MarketReports/SwapsReports/Archive/index.htm

## Table: `cftc.swaps.weekly`

### Schema

- `time_column`: `date` (publication date or week ending; choose one and be consistent)
- `entity_column`: `entity_id`

Required columns (minimum):
- `date`
- `entity_id`
- `asof_utc` (publication timestamp if available; else `date` at 00:00 UTC)
- `value` (float)

Canonical columns:
- `value`
- plus metadata columns when available:
  - `report_name`
  - `metric`
  - `currency` (if present)
  - `maturity_bucket` (if present)
  - `participant_type` (if present)

### Entity IDs

Encode the dimensionality into entity_id:

Example pattern:
- `cftc.swaps.rates.<currency>.<bucket>.<metric>.<participant>`
- if currency not present, use `all`

### Acquisition

- crawler visits archive page
- extracts links to weekly XLSX
- downloads new files only (cache by URL hash)
- parses each sheet per known layout (use header detection; avoid hard-coded row numbers where possible)

### Fetch

- filter by date range and entities
- column pushdown: always return `value` plus requested cols; but simplest is to offer `value` and a few metadata columns.

### Tests

- fixture: one archived XLSX saved locally
- parser test: extracts at least N series, stable entity IDs
