# Dataset Spec: CME ProductSlate (Reference Data)

## Source

Direct CSV download:

URL:
- https://www.cmegroup.com/CmeWS/mvc/ProductSlate/V1/Download.csv

## Table: `cme.productslate.reference`

### Purpose

Reference data for CME products (futures/options), used for:
- mapping symbols to product families
- identifying rate vs FX vs equity products
- expiry / product metadata

### Schema

- `time_column`: `date` (snapshot date)
- `entity_column`: `entity_id` (use CME product code or a stable identifier)

Required columns:
- `date`
- `entity_id`
- `asof_utc`
- `exchange` (string)
- `product_code` (string)
- `product_name` (string)
- `asset_class` (string) if present
- `sub_asset_class` (string) if present

Canonical columns:
- `clearing_code`, `globex_symbol`, `mic`, etc. if present

### Acquisition

- download CSV daily (or on-demand)
- cache by content hash; update only when changed

### Tests

- fixture: a sample CSV subset
- parser test: required columns exist; entity IDs stable
