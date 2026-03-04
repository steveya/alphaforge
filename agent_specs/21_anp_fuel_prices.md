# Spec 21 — ANP Fuel Price Survey (Brazil, weekly)

## Source
Brazil government open data for ANP fuel prices.
Dataset landing:
- https://dados.gov.br/dados/conjuntos-dados/serie-historica-de-precos-de-combustiveis-e-de-glp
ANP page:
- https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis

## Purpose (inflation nowcasting)
Weekly retail fuel prices by geography (high-frequency energy input).

## Implementation
### File
`alphaforge/data/public_web/anp_fuel_prices.py`

### Class
`ANPFuelPricesDataSource`

## Discovery / manifest
The dataset page contains one or more downloadable CSVs (sometimes many).
Implement:
1) fetch HTML
2) extract all CSV (and ZIP) download URLs
3) maintain a manifest cache (reuse existing caching utilities)

Support a configuration:
- `mode="latest_only"` default to reduce load
- `mode="all"` to backfill entire history

## Table
`anp_fuel_prices_weekly`
- `entity_id` = `ANP_FUEL:{product}:{geo_level}:{geo_code}`
- `date` = week reference date (from CSV)
- `value` = average price (float)
- `asof_utc`
Optional columns:
- `min_price`, `max_price`, `num_stations`, `unit` (only if cleanly available)

## Parsing
- Handle decimal separators and locale formats.
- Identify fuel/product field, geography fields (state/municipality/region), and date field.
- Normalize to tidy long format.

## Query mapping
- `q.entities` optional: filter entity_id
- `q.start/q.end`: filter by date

## Tests
`tests/test_public_web_anp_fuel_prices.py`
- Load latest file and assert non-empty.
- Filter to one entity_id present and assert non-empty.