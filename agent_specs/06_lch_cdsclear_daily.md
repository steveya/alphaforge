# Dataset Spec: LCH CDSClear Volumes (Daily)

## Source

LCH CDSClear volume page provides updated daily statistics.

URL:
- https://www.lseg.com/en/post-trade/clearing/lch-services/cdsclear/volumes

## Table: `lch.cdsclear.daily`

### Schema

- `time_column`: `date` (UTC midnight)
- `entity_column`: `entity_id`

Required columns:
- `date`
- `entity_id`
- `asof_utc`
- `value` (float)

Canonical columns:
- include `metric` (e.g., notional, trades) if more than one
- include region/segment if present (e.g., iTraxx, CDX, sovereign)

### Entity IDs

Example:
- `lch.cdsclear.volume.cleared_notional`
- `lch.cdsclear.volume.trades`

If the page breaks out by index family, include it:
- `lch.cdsclear.volume.itraxx.cleared_notional`

### Acquisition

- scrape HTML tables
- parse date labels and values

### Tests

- fixture: saved HTML snapshot
- parser test: stable extraction by header matching
