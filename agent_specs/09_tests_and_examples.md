# Tests, CI, and Example Usage

## Test layout

Add:

- `tests/public_web/test_dtcc_ppd.py`
- `tests/public_web/test_cftc_swaps_weekly.py`
- `tests/public_web/test_eurex_stats_daily.py`
- `tests/public_web/test_eurex_refdata_contracts.py`
- `tests/public_web/test_lch_cdsclear_daily.py`
- `tests/public_web/test_cme_productslate_reference.py`
- `tests/public_web/test_ezoic_adrevenue_daily.py`

Fixtures:

- `tests/fixtures/public_web/dtcc_ppd/sample.zip`
- `tests/fixtures/public_web/cftc_swaps/sample.xlsx`
- `tests/fixtures/public_web/eurex_market_stats/sample.html`
- `tests/fixtures/public_web/eurex_refdata/sample.json`
- `tests/fixtures/public_web/lch_cdsclear/sample.html`
- `tests/fixtures/public_web/cme_productslate/sample.csv`
- `tests/fixtures/public_web/ezoic/sample.html` or `sample.json`

## Contract tests template

For each table:
- `source.schemas()[table].required_columns` must exist in output
- output includes `entity_column` and `time_column`
- timestamps are tz-aware UTC
- `fetch()` honors `Query.start/end/entities` for at least one fixture

## Live tests (optional)

Mark with `pytest.mark.network` and skip unless:
- `ALPHAFORGE_NETWORK_TESTS=1`

## Example usage

Add `examples/public_web_macro_derivs.py`:

- instantiate sources:
  - `DTCCPPDSource(cache_dir=...)`
  - etc.
- create a `DataContext(sources={source.name: source}, calendars={}, store=...)`
- fetch:
  - `dtcc.ppd.daily` for last 90 days for a few entities
  - `eurex.stats.daily` for bund volume/OI
  - `ezoic.adrevenue.daily` as risk overlay
- print head; optionally materialize to store
