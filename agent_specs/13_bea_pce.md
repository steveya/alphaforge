# Spec 13 — BEA PCE Price Indices (US)

## Source
BEA API (free key required).
Docs:
- https://apps.bea.gov/api/_pdf/bea_web_service_api_user_guide.pdf

## Purpose (inflation nowcasting)
Pull:
- PCE price index
- Core PCE price index
Optionally PCE component deflators.

## Implementation
### File
`alphaforge/data/public_web/bea.py`

### Class
`BEADataSource`

### Auth
API key required: env `BEA_API_KEY` or constructor arg.

### Endpoint
GET:
- `https://apps.bea.gov/api/data`

Common params:
- `UserID={key}`
- `method=GetData`
- `datasetname=...`
- plus dataset-specific parameters.

## Registry (required)
BEA “series” is a parameter bundle. Create:
- `alphaforge/data/registries/bea_series.yaml`

Each entry:
```yaml
- entity_id: "BEA:NIPA:T20804:L2:M:PCEPI"
  description: "PCE price index (example—agent must confirm exact table/line)"
  params:
    datasetname: "NIPA"
    TableName: "T20804"
    LineNumber: "2"
    Frequency: "M"
    Year: "ALL"
```
Agent must populate at minimum two entities:
- headline PCE price index
- core PCE price index
(Use correct BEA table/line codes from BEA docs / tables.)

## Table
`bea_series`
- `entity_id` = registry key
- `date` = month/quarter end UTC
- `value` = float
- `asof_utc` = UTC timestamp

## Query mapping
- `q.entities`: list of `entity_id` registry keys.
- `q.start/q.end`: filter output by date. For BEA request, it’s okay to request Year=ALL then filter client-side, or map to years.
- `q.columns`: `value` only.

## Parsing
BEA returns `TimePeriod` and `DataValue`.
Convert:
- "YYYY-MM" → month end UTC
- "YYYYQ1"…"YYYYQ4" → quarter end UTC
- "YYYY" → year end UTC

Normalize missing/“(NA)” to NaN.

## Tests
`tests/test_public_web_bea.py`
- Use a single BEA entity from registry (headline PCEPI) for 2018–2020.
- Assert required columns and date range filtering.