# Quickstart: Public Web Loaders

Alphaforge ships a public loader pack under `alphaforge.data.public_web` for
public datasets that need stable schemas but not a full custom adapter stack.

These loaders currently sit on the `DataSource` side of the compatibility
boundary. They are appropriate when you want raw long-form frames from the
public-web pack directly, but they are not the canonical general-purpose
loading path for new adapter-based integrations.

## Loader families

- Registry-backed APIs: `bcb_sgs_series`, `bea_series`, `bls_series`,
  `destatis_series`, `ecb_sdmx_series`, `eia_series`, `eurostat_series`,
  `ibge_sidra_series`
- Tabular and document loaders: `cme.productslate.reference`,
  `ec_oil_bulletin_weekly`, `eurex.refdata.contracts`, `eurex.stats.daily`,
  `ezoic.adrevenue.daily`, `frb.term_structure`, `lch.cdsclear.daily`
- Archive and batch loaders: `b3_equity_quotes_daily`, `cftc.cot.tff`,
  `cftc.cot.disagg`, `cftc.swaps.weekly`
- Provider-specific outliers: `anp_fuel_prices_weekly`, `dtcc.ppd.events`,
  `dtcc.ppd.daily`, `mof.jgb.yields`, `philadelphia.spf.mean_level`

## Minimal usage

This quickstart intentionally uses the raw loader objects directly:

```python
import pandas as pd

from alphaforge import default_public_web_sources
from alphaforge.data.query import Query

sources = default_public_web_sources()

dtcc = sources["dtcc_ppd"]
cot = sources["cftc_cot_disagg"]

dtcc_daily = dtcc.fetch(
    Query(
        table="dtcc.ppd.daily",
        columns=["trade_count", "notional_sum", "price_std"],
        start=pd.Timestamp("2026-01-01", tz="UTC"),
        end=pd.Timestamp("2026-01-31", tz="UTC"),
    )
)

cot_frame = cot.fetch(
    Query(
        table="cftc.cot.disagg",
        columns=["value"],
        entities=["wheat_srw"],
        start=pd.Timestamp("2025-01-01", tz="UTC"),
    )
)
```

If you instantiate a source directly instead of using
`default_public_web_sources()`, keep constructor-specific requirements explicit.
For example, `EurexRefdataContractsSource` requires an `api_url`.

If you are building a broader application-level loading layer, prefer wrapping
that usage behind `SourceAdapter` plus `DataContext.fetch(...)` rather than
treating direct `DataSource.fetch(...)` calls as the primary public contract.

## Validation workflow

Fast local checks:

```bash
/Users/steveyang/miniforge3/bin/python -m pytest tests/public_web -k 'not live_sources'
/Users/steveyang/miniforge3/bin/python -m ruff check .
```

Optional live tests:

```bash
ALPHAFORGE_NETWORK_TESTS=1 /Users/steveyang/miniforge3/bin/python -m pytest \
  tests/public_web/test_live_sources.py -q
```

For contributor guidance on adding or refactoring loaders, see the
[public-web source authoring guide](../guides/public-web-source-authoring.md).
