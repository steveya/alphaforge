# Dataset Spec: Eurex Free Reference Data API

## Source

Eurex provides a free reference data API.

URL:
- https://www.eurex.com/ex-en/data/free-reference-data-api

## Table: `eurex.refdata.contracts`

### Goal

Provide a contracts master table to support:
- mapping contract codes to product families
- maturity information
- currency and underlying type

### Schema

- `time_column`: `date` (set to ingestion date)
- `entity_column`: `entity_id` (contract identifier)

Required columns:
- `date`
- `entity_id` (e.g., Eurex contract id)
- `asof_utc`
- `symbol`
- `product_name`
- `product_group`
- `currency`
- `expiry_date` (UTC date)

Canonical columns:
- `underlying`, `multiplier`, `tick_size`, `isin` (if available)

### Acquisition

- call the free reference data API endpoint(s) documented on the page
- cache responses by day

### Fetch

- treat this as slowly changing reference data; filter by q.entities when specified
- ignore q.start/q.end unless you store historical snapshots; simplest: return latest snapshot and set `date` to snapshot date.

### Tests

- fixture: sample JSON response saved locally
- parser test: contract rows count > 0, required columns exist
