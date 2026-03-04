# Spec 18 — European Commission Weekly Oil Bulletin (EU fuel prices)

## Source
Weekly Oil Bulletin page:
- https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en
Raw data (history):
- https://energy.ec.europa.eu/publications/oil-bulletin-raw-data_en

## Purpose (inflation nowcasting)
Weekly retail fuel prices by country (high-frequency energy input).

## Implementation
### File
`alphaforge/data/public_web/ec_weekly_oil_bulletin.py`

### Class
`ECWeeklyOilBulletinDataSource`

## Discovery
Implement HTML scraping of the Weekly Oil Bulletin page to locate download links:
- “prices with taxes” XLSX
- “prices without taxes” XLSX
Optionally scrape raw-data archive for historical coverage.

Cache discovered URLs using existing cache utilities to avoid frequent HTML parsing.

## Table
`ec_oil_bulletin_weekly`
- `entity_id` = `EC_OIL:{product}:{country_code}:{tax_flag}`
  - product examples: `EUROSUPER95`, `DIESEL`
  - country_code: ISO2 (DE, FR, ...)
  - tax_flag: `WITH_TAX` or `NO_TAX`
- `date` = bulletin week reference date (use explicit date from sheet if present; else from page label)
- `value` = price (float)
- `asof_utc`

## Parsing
- Download XLSX bytes
- Parse with pandas `read_excel`
- Identify columns for country/product/price
- Normalize decimals and missing values

## Query mapping
- `q.entities` optional: filter to subset of `entity_id`
- `q.start/q.end`: date filter

## Tests
`tests/test_public_web_ec_oil_bulletin.py`
- Fetch latest with no entities; assert some countries exist.
- Fetch one specific entity_id and assert non-empty.