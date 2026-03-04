# Spec 19 — IBGE SIDRA (Brazil)

## Source
SIDRA API base:
- http://api.sidra.ibge.gov.br/

Primary reference page (IPCA):
- https://www.ibge.gov.br/en/statistics/economic/prices-and-costs/17129-extended-national-consumer-price-index.html

## Purpose (inflation nowcasting)
Brazil IPCA inflation and related indicators accessible via SIDRA tables.

## Implementation
### File
`alphaforge/data/public_web/ibge_sidra.py`

### Class
`IBGESidraDataSource`

## Registry (required)
Create:
- `alphaforge/data/registries/ibge_sidra_series.yaml`

Entry structure:
```yaml
- entity_id: "IBGE:IPCA:ALL_ITEMS:IDX"
  description: "IPCA all items index or rate (agent to decide and document)"
  table: "XXXX"
  variable: "YYY"
  period: "all"
  params:
    format: "json"
    geo: "1"
```
Agent must identify the correct SIDRA table/variable for IPCA headline (index or monthly rate).

## Endpoint strategy
SIDRA uses structured paths for table queries. Implement:
- build URL from registry entry (table/variable/period/geo/classifications)
- request JSON
- parse (period, value)

## Table
`ibge_sidra_series`
- `entity_id`
- `date` = month end UTC
- `value`
- `asof_utc`

## Tests
`tests/test_public_web_ibge_sidra.py`
- One IPCA entity, 2018–2020.