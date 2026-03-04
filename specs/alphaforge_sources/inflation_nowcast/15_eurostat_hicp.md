# Spec 15 — Eurostat HICP (EU / EA)

## Source
Eurostat dissemination APIs (SDMX / JSON-stat) and guides.
Docs:
- https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-introduction
- https://ec.europa.eu/eurostat/api/dissemination/swagger-ui

## Purpose (inflation nowcasting)
Pull HICP (headline and optionally core / components) for:
- Euro area aggregate
- individual member states

## Implementation
### File
`alphaforge/data/public_web/eurostat.py`

### Class
`EurostatDataSource`

## Registry (required)
Eurostat requires dataset id + dimensional filters. Create:
- `alphaforge/data/registries/eurostat_series.yaml`

Entry structure:
```yaml
- entity_id: "EUROSTAT:HICP:EA:CP00:IDX"
  description: "Euro area HICP all-items index (example)"
  dataset: "prc_hicp_midx"  # example; agent must confirm correct dataset id(s)
  filters:
    geo: "EA20"
    coicop: "CP00"
    unit: "I15"
    freq: "M"
```
Agent must populate at minimum:
- EA headline HICP series (index or yoy rate—pick one and document)
- DE/FR/IT/ES equivalents for testing

## Endpoint strategy
Prefer an endpoint returning JSON-stat or CSV to avoid SDMX-ML parsing.
Implementation should:
- build request URL from `dataset` + `filters`
- fetch payload with shared http helper
- parse into (time, value)

## Table
`eurostat_series`
- `entity_id`
- `date` = month end UTC
- `value`
- `asof_utc`

## Date parsing
Eurostat monthly time keys may appear as "YYYYMmm", "YYYY-MM", or "YYYY-MM-01".
Normalize to month end UTC.

## Tests
`tests/test_public_web_eurostat.py`
- Use one EA HICP series for 2018–2020.
- Assert monotonic monthly dates and correct filtering.