# Dataset Spec: Eurex Market Statistics Online (Daily)

## Source

Eurex provides daily market statistics via web tables.

URL:
- https://www.eurex.com/ex-en/data/statistics/market-statistics-online

## Table: `eurex.stats.daily`

### Goal

Provide daily time series for:
- volume
- open interest
by major product groups relevant to rates and risk (rates, equity index vol, etc.).

### Schema

- `time_column`: `date` (UTC midnight)
- `entity_column`: `entity_id`

Required columns:
- `date`
- `entity_id`
- `asof_utc` (ingestion timestamp)
- `volume` (float)
- `open_interest` (float) when available

Canonical columns:
- add `contract_count` or `trades` if present
- add `product_group`, `product_name` (string) if returned

### Entity IDs

At minimum:
- `eurex.<product_group>.<product_or_family>.<metric>`

Example:
- `eurex.rates.bund_fut.volume`
- `eurex.rates.bund_fut.open_interest`

### Acquisition approach

Option A (simpler):
- daily scrape the HTML table(s) and parse with pandas `read_html`
- identify tables by headers (e.g., contains "Open Interest" and "Volume")
- normalize numbers (remove commas, handle missing)

Option B (better if Eurex exposes a downloadable file):
- if the page links to CSV/Excel, prefer that (agent should check page structure).

### Fetch behavior

- since this is daily, filter on `date` with q.start/q.end
- `q.entities` selects entity IDs.

### Tests

- fixture: saved HTML page (snapshot)
- parser test ensures stable extraction even if table order changes (use header matching)
