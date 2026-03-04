# Spec 17 — Destatis GENESIS (Germany CPI/HICP)

## Source
Destatis GENESIS-Online API.
Docs:
- https://www.destatis.de/EN/Service/OpenData/api-webservice.html

## Purpose (inflation nowcasting)
Germany official CPI/HICP series (monthly).

## Implementation
### File
`alphaforge/data/public_web/destatis_genesis.py`

### Class
`DestatisGenesisDataSource`

## Auth
GENESIS typically requires credentials / token. Implement:
- env `DESTATIS_GENESIS_USER` / `DESTATIS_GENESIS_PASS` OR `DESTATIS_GENESIS_KEY`
Use what your existing helper/registry expects; if unknown, implement constructor accepting a dict and document.

## Registry (required)
Create:
- `alphaforge/data/registries/destatis_series.yaml`

Entry structure (example; agent must fill real codes):
```yaml
- entity_id: "DESTATIS:CPI:ALL_ITEMS:IDX"
  description: "Germany CPI all items index"
  table_code: "61111-0001"
  params:
    format: "json"
```
Populate at minimum:
- CPI all-items
- HICP all-items

## Endpoint strategy
GENESIS has specific endpoints and parameterization; implement a thin client that:
- requests a table/variable/time via GENESIS API
- parses returned JSON/CSV into (period, value)

## Table
`destatis_series`
- `entity_id`
- `date` = month end UTC
- `value`
- `asof_utc`

## Tests
`tests/test_public_web_destatis.py`
- One CPI series, 2018–2020.
- If auth required, allow skipping when credentials absent.