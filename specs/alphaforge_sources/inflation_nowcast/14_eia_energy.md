# Spec 14 — EIA Energy Prices (US)

## Source
US EIA Open Data API v2.
Docs:
- https://www.eia.gov/opendata/documentation.php
- https://www.eia.gov/opendata/documentation/APIv2.1.0.pdf

## Purpose (inflation nowcasting)
High-frequency energy inputs:
- Daily crude oil price series
- Weekly retail gasoline prices
Optional: diesel, heating oil, natural gas.

## Implementation
### File
`alphaforge/data/public_web/eia.py`

### Class
`EIADataSource`

### Auth
API key required: env `EIA_API_KEY` or constructor arg.

## Registry (required)
EIA v2 queries require route + facets bundle. Create:
- `alphaforge/data/registries/eia_series.yaml`

Entry structure:
```yaml
- entity_id: "EIA:CRUDE:WTI:SPOT_D"
  description: "WTI spot price, daily (example; agent must fill correct route/facets)"
  route: "petroleum/pri/spt"
  params:
    frequency: "daily"
    data: ["value"]
    facets: {}
```
Agent must populate at minimum:
- one daily crude oil price series
- one weekly US gasoline series

## Endpoint pattern
Base:
- `https://api.eia.gov/v2/{route}/data/`

Query uses:
- `api_key=...`
- `frequency=...`
- `data[0]=...`
- `facets[... ][]=...`
- `start=YYYY-MM-DD` / `end=YYYY-MM-DD` if supported, else client-side filtering.

## Table
`eia_series`
- `entity_id` = registry key
- `date` = observation timestamp (daily/weekly/monthly)
- `value` = float
- `asof_utc`

Weekly date convention:
- If API returns an explicit period date (preferred), use that.
- Otherwise, treat as week-ending date and document it in registry `notes`.

## Query mapping
- `q.entities`: list of registry `entity_id`
- `q.start/q.end`: pass through to API when supported; always filter client-side as final step.

## Tests
`tests/test_public_web_eia.py`
- One daily series and one weekly series, small historical window.