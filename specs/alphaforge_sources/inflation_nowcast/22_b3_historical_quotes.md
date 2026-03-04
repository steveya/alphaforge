# Spec 22 — B3 Historical Quote Data (Brazil, public file downloads; optional)

## Source
B3 historical quote data (equities) page:
- https://www.b3.com.br/en_us/market-data-and-indices/data-services/market-data/historical-data/equities/historical-quote-data/

## Purpose (inflation nowcasting)
Optional macro-financial predictors (equity prices/returns) used in some nowcasting setups.
Not strictly required for CPI/IPCA/HICP nowcasts, but useful for “macro-financial indicator” sets.

## Implementation
### File
`alphaforge/data/public_web/b3_historical_quotes.py`

### Class
`B3HistoricalQuotesDataSource`

## Discovery
Scrape the B3 page for download links (often annual ZIP/TXT files).
Use q.start/q.end to decide which years to download.
Cache downloads locally (reuse existing cache utilities).

## Parsing
B3 historical quote files can be fixed-width with a known layout. Implement a minimal parser:
- date
- ticker
- open/high/low/close
- volume
Normalize numeric scaling as documented by B3 (agent to confirm from file layout notes).

## Table
`b3_equity_quotes_daily`
- `ticker`
- `date` (UTC)
- `open`, `high`, `low`, `close`, `volume`
- `asof_utc`

## Tests
`tests/test_public_web_b3_quotes.py`
- Download one small year file (or a small subset) and parse a handful of rows.
- Mark test as “slow/network” if needed.