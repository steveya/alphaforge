# Spec 16 — ECB SDMX (HICP via ECB Data Portal)

## Source
ECB SDMX 2.1 REST service docs:
- https://data.ecb.europa.eu/help/api/overview
- https://data.ecb.europa.eu/help/api/data

## Purpose (inflation nowcasting)
Alternative path to obtain HICP series via ECB SDMX service (useful when you have SDMX keys).

## Implementation
### File
`alphaforge/data/public_web/ecb_sdmx.py`

### Class
`ECBSDMXDataSource`

## Registry (required)
Create:
- `alphaforge/data/registries/ecb_sdmx_series.yaml`

Entry structure:
```yaml
- entity_id: "ECB:HICP:EA:CP00:IDX"
  description: "EA HICP series (example; agent must set real flowRef/key)"
  flowRef: "HICP"
  key: "M.U2.N.000000.4.INX"    # placeholder
  params:
    format: "csvfile"
```
Agent must populate at minimum one HICP series that returns successfully.

## Endpoint
Typical ECB SDMX REST:
- `https://data-api.ecb.europa.eu/service/data/{flowRef}/{key}?startPeriod=...&endPeriod=...`

Prefer CSV output if supported; otherwise parse SDMX-ML/JSON.

## Table
`ecb_sdmx_series`
- `entity_id`
- `date` = month end UTC
- `value`
- `asof_utc`

## Query mapping
- `q.entities`: registry entity_id list
- `q.start/q.end`: translate to `startPeriod/endPeriod` (YYYY-MM)

## Tests
`tests/test_public_web_ecb_sdmx.py`
- One HICP series, 2019–2020.
- Mark as optional/xfail if SDMX key becomes invalid; prefer Eurostat as primary.