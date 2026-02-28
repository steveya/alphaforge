# Quickstart: Public Web Loaders

Alphaforge includes a public web loader pack under `alphaforge.data.public_web`.

## Supported table families

- `dtcc.ppd.events`
- `dtcc.ppd.daily`
- `cftc.swaps.weekly`
- `eurex.stats.daily`
- `eurex.refdata.contracts`
- `lch.cdsclear.daily`
- `cme.productslate.reference`
- `ezoic.adrevenue.daily`

## Minimal usage

```python
import pandas as pd

from alphaforge.data.query import Query
from alphaforge.data.public_web import DTCCPPDSource, EurexStatsDailySource

dtcc = DTCCPPDSource(artifact_provider=...)
eurex = EurexStatsDailySource()

dtcc_daily = dtcc.fetch(
    Query(
        table="dtcc.ppd.daily",
        columns=["trade_count", "notional_sum", "price_std"],
        start=pd.Timestamp("2026-01-01", tz="UTC"),
        end=pd.Timestamp("2026-01-31", tz="UTC"),
    )
)

eurex_stats = eurex.fetch(
    Query(
        table="eurex.stats.daily",
        columns=["volume", "open_interest"],
    )
)
```

## Testing loaders

```bash
pytest -o addopts='' tests/public_web
```

Optional live tests:

```bash
ALPHAFORGE_NETWORK_TESTS=1 pytest -o addopts='' tests/public_web/test_live_sources.py
```
