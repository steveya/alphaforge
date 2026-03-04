# Spec 20 — Banco Central do Brasil SGS (Brazil)

## Source
BCB SGS time series API (public).
Common pattern:
- `http://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados?formato=csv&dataInicial=DD/MM/YYYY&dataFinal=DD/MM/YYYY`

Open data portal:
- https://opendata.bcb.gov.br/

## Purpose (inflation nowcasting)
Macro/financial predictors: policy rates, FX, etc. (public time series codes).

## Implementation
### File
`alphaforge/data/public_web/bcb_sgs.py`

### Class
`BCBSGSDataSource`

## Table
`bcb_sgs_series`
- `entity_id` = SGS code as string
- `date` = provider date (daily/monthly depending on series)
- `value` float
- `asof_utc`

## Query mapping
- `q.entities`: list of SGS codes
- `q.start/q.end`: translate to `dataInicial/dataFinal` in DD/MM/YYYY
- `formato=csv` preferred; parse CSV robustly

## Notes
Some SGS series are daily, some monthly. Preserve provider dates.

## Tests
`tests/test_public_web_bcb_sgs.py`
- Pick one stable SGS series id (agent to choose) and a historical window.
- Assert required columns and proper date parsing.