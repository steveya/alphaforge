# Spec 12 — BLS CPI (US)

## Source
US Bureau of Labor Statistics (BLS) Public Data API v2.
Docs:
- https://www.bls.gov/bls/api_features.htm
- https://www.bls.gov/developers/api_signature_v2.htm

## Purpose (inflation nowcasting)
Pull CPI-U headline, core, and selected CPI components (monthly).

## Implementation
### File
`alphaforge/data/public_web/bls.py`

### Class
`BLSDataSource`

### Auth
Support:
- optional API key (recommended) via env `BLS_API_KEY` or constructor arg.
If no key, still attempt request (BLS may allow limited access).

### Endpoint
POST:
- `https://api.bls.gov/publicAPI/v2/timeseries/data/`

Request body (JSON) example:
```json
{
  "seriesid": ["CUUR0000SA0", "CUUR0000SA0L1E"],
  "startyear": "2020",
  "endyear": "2026",
  "registrationkey": "...."
}
```

### Table
`bls_series`
- `entity_id` = BLS series id (string)
- `date` = month end UTC
- `value` = float
- `asof_utc` = UTC timestamp

### Query mapping
- `q.entities`: list of BLS series ids.
- `q.start/q.end`: map to `startyear/endyear` and then filter precisely client-side by month.
- `q.columns`: only `value` is guaranteed.

### Parsing rules
BLS returns year + period "M01"..."M12".
Convert to month-end UTC:
- `date = Timestamp(year, month, 1, tz=UTC) + MonthEnd(0)`

Handle missing values as NaN.

### Chunking
BLS accepts multiple series ids; implement chunking (e.g., 25–50 per request) if `q.entities` is large.

## Suggested registry
Optional (not required) if you want human-friendly aliases:
- `alphaforge/data/registries/bls_aliases.yaml` mapping alias → seriesid.

## Tests
`tests/test_public_web_bls.py`
- Use one stable CPI series (headline CPI-U) for a fixed window (e.g., 2019-01 to 2021-12).
- Assert:
  - required columns
  - monthly dates monotonic increasing
  - numeric parse