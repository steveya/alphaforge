# Source Adapters

Canonical unified data-loading surface for Alphaforge.

New code should register `SourceAdapter` implementations in
`DataContext.adapters` and load data through:

- `DataContext.fetch(...)`
- `DataContext.fetch_many(...)`
- `DataContext.prefetch(...)`

Legacy `DataSource` and `fetch_panel(...)` usage remains supported only as a
compatibility boundary while older loaders migrate.

## Protocol & Base

::: alphaforge.data.adapter

## Value Types

::: alphaforge.data.types

## Cache Layer

::: alphaforge.data.cache_layer

## Discovery

::: alphaforge.data.sources.discover_adapters

## Built-in Adapters

### Tiingo (Market OHLCV)

::: alphaforge.data.sources.tiingo.TiingoAdapter

### FRED (Macro PIT)

::: alphaforge.data.sources.fred.FREDSourceAdapter

### CFTC (CoT Positioning)

::: alphaforge.data.sources.cftc.CFTCAdapter

### DTCC (Swap Derivatives)

The preferred DTCC adapter-backed datasets now split along the first concrete
product families:

- `DTCCFXAdapter` serves `dtcc.fx` for FX forwards and swaps
- `DTCCIRSAdapter` serves `dtcc.irs` for interest rate swaps

These family adapters own `DTCCPPDSource` construction internally, stamp
product-family-specific PIT lineage, and keep dataset routing distinct in
`DataContext`.

```python
from alphaforge.data.context import DataContext
from alphaforge.data.sources.dtcc import DTCCFXAdapter, DTCCIRSAdapter

ctx = DataContext.from_adapters(
    DTCCFXAdapter(list_provider=..., artifact_provider=...),
    DTCCIRSAdapter(list_provider=..., artifact_provider=...),
)

fx = ctx.load(
    "dtcc.fx",
    columns=["value"],
    entities=["dtcc.fx.dtccppd.fx.fx_forward.usd.1m.trade_count"],
)
irs = ctx.load(
    "dtcc.irs",
    columns=["value"],
    entities=["dtcc.irs.dtccppd.rates.interest_rate_swap.usd.5y.trade_count"],
)
```

`DTCCAdapter` remains available as the broader generic `dtcc.ppd` wrapper, and
future DTCC product-family adapters should build on `DTCCPPDAdapterBase` so
raw-loader fetch wiring, cache behavior, and PIT transform plumbing stay
consistent.

::: alphaforge.data.sources.dtcc.DTCCFXAdapter

::: alphaforge.data.sources.dtcc.DTCCIRSAdapter

::: alphaforge.data.sources.dtcc.DTCCAdapter

::: alphaforge.data.sources.dtcc.DTCCPPDAdapterBase

## PIT Compatibility Bridge

`SourceAdapterPITCompat` is a temporary migration bridge. It remains documented
because supported downstream PIT integrations still rely on it, but it is not
the long-term canonical adapter model.

::: alphaforge.pit.adapters.source_adapter_compat.SourceAdapterPITCompat
