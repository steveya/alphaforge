# Entity ID Taxonomy (Alphaforge Public-Web Loaders)

This document defines a **single, consistent `entity_id` taxonomy** used across all public-web loaders
in this pack, including DTCC PPD event-level and daily aggregates.

Goals:
- Stable, parseable entity IDs
- Consistent across sources (DTCC, Eurex, LCH, CFTC, CME, Ezoic)
- Minimal yet expressive enough for macro bucketing (G6 rates / G10 FX / cross-asset risk)

---

## 1. General format

All entity IDs are **dot-separated**, lowercase ASCII:

```
<domain>.<asset>.<instrument>.<ccy_or_pair>.<tenor_or_term>.<metric>[.<source>][.<qualifiers...>]
```

### Mandatory segments by table type

**Event-level trade/print data** (e.g., `dtcc.ppd.events`):
- domain, asset, instrument, ccy_or_pair, tenor_or_term are required
- metric is optional at event-level (because events already have multiple fields)
- include `source` as a suffix segment when useful for disambiguation (recommended)

**Panel metrics data** (daily/weekly aggregates, stats, indices):
- metric is required
- include `source` when multiple sources provide similar entities

### Character rules

- Only `[a-z0-9_.-]`
- No spaces
- Replace `/` in FX pairs with nothing (e.g., `eurusd`, not `eur/usd`)
- Use `_` for multiword product names (e.g., `bund_fut`)

---

## 2. Standard namespaces

### 2.1 Domains

- `rates`     : interest rates (cash curves, swaps, futures, options)
- `fx`        : foreign exchange (spot, forwards, swaps, options)
- `credit`    : CDS indices/options proxies
- `risk`      : cross-asset risk proxies (vol indices, breadth, etc.)
- `macro`     : macro alternative indicators (e.g., ad revenue index)

### 2.2 Asset segment (coarse)

- `irs`       : interest rate swaps (incl OIS/vanilla IRS depending on instrument)
- `ois`       : overnight index swaps (explicitly)
- `swaption`  : swaptions (OTC or listed, depending on source)
- `fut`       : futures
- `opt`       : options on futures
- `spot`      : FX spot
- `fwd`       : FX forwards
- `xcs`       : FX cross-currency swaps / basis swaps
- `cds`       : credit default swaps
- `index`     : index-like series

### 2.3 Instrument segment

For rates:
- `irs` / `ois` / `swaption` / `fut` / `opt`

For FX:
- `spot` / `fwd` / `xcs` / `opt`

For cross-asset:
- `index` / `opt` (when referring to volatility/option proxies)

> Note: For rates, `asset` and `instrument` are often identical; keep both for consistency and allow future expansion.

---

## 3. Currency / Pair segment

### 3.1 Rates currency codes

Use ISO-ish currency codes:
- `usd, eur, jpy, gbp, chf, cad, aud, nzd, sek, nok`

### 3.2 FX pairs

Concatenate base+quote lowercase:
- `eurusd, usdjpy, gbpusd, audjpy, eurnok, eursek, usdcad, audusd`, etc.

If you need crosses with nonstandard quoting, preserve market convention but still concatenate.

---

## 4. Tenor / Term segment

### 4.1 Standard tenor buckets

Use one of:

- money market: `1w, 2w, 1m, 2m, 3m, 6m, 9m`
- yearly: `1y, 2y, 3y, 4y, 5y, 6y, 7y, 8y, 9y, 10y`
- long end: `12y, 15y, 20y, 25y, 30y, 40y, 50y`

If unknown:
- `unk`

### 4.2 FX forward tenors

Use same tenor bucket list:
- `1w, 1m, 3m, 6m, 1y`, etc.
If a date-range is given without tenor, map to nearest.

### 4.3 Futures/options expiry term

For listed contracts where expiry matters, use:
- `yyyymm` (e.g., `202606`) for contract month
- or `front`, `next`, `back` when you roll compressively

Recommended:
- keep expiry-specific entities only in **reference tables**.
- for daily stats tables, prefer bucketing to `front` / `next` / `pack` rather than thousands of expiries.

---

## 5. Metric segment (panel series)

Standard metrics:
- `value`               : generic scalar index
- `volume`              : traded volume
- `open_interest`       : open interest
- `trade_count`         : number of prints/trades
- `notional_sum`        : sum of reported notionals
- `price_mean`          : mean executed price/rate
- `price_std`           : dispersion
- `price_p10`, `price_p50`, `price_p90`
- `dv01_proxy_sum`      : derived proxy
- `cleared_share`       : share cleared
- `sef_share`           : share on SEF/venue (when available)

---

## 6. Source segment (optional but recommended)

Use a short, stable identifier:

- `dtccppd`     : DTCC PPD (CFTC dissemination)
- `cftc`        : CFTC weekly swaps report
- `eurex`       : Eurex exchange stats
- `lch`         : LCH CDSClear volumes
- `cme`         : CME reference data
- `ezoic`       : Ezoic ad revenue index

If included, put it **as the last segment**:
- `rates.ois.eur.2y.trade_count.dtccppd`

---

## 7. Table-specific prescriptions

### 7.1 `dtcc.ppd.events`

**Event-level**: `entity_id` encodes instrument + currency + tenor bucket.

Template:
```
rates.<instrument>.<ccy>.<tenor>.dtccppd
fx.<instrument>.<pair>.<tenor>.dtccppd
```

Examples:
- `rates.ois.eur.2y.dtccppd`
- `rates.irs.gbp.5y.dtccppd`
- `rates.swaption.usd.1y.dtccppd`  (if swaptions disseminated with tenor)
- `fx.fwd.usdjpy.3m.dtccppd`
- `fx.opt.eurusd.1m.dtccppd`       (only if option product fields support it)

If missing tenor:
- `rates.ois.eur.unk.dtccppd`

If missing currency:
- `rates.ois.unk.unk.dtccppd`

### 7.2 `dtcc.ppd.daily`

**Derived**: same `entity_id` as events. Metric is in columns, not in entity_id.

If you prefer “metric-as-entity”, you may ALSO expose a second view in future; for this pack keep:
- entities = instrument/currency/tenor
- metrics = columns

### 7.3 `cftc.swaps.weekly`

Template:
```
rates.<instrument>.<ccy>.<bucket>.<metric>.cftc
```

Examples:
- `rates.irs.usd.2y5y.trade_count.cftc`
- `rates.irs.usd.10yplus.notional_sum.cftc`

If the report provides participant types:
- add qualifier at end:
  - `.dealer`, `.asset_manager`, `.leveraged`, etc.
Example:
- `rates.irs.usd.2y5y.notional_sum.cftc.dealer`

### 7.4 `eurex.stats.daily`

Template:
```
rates.fut.<symbol_family>.<metric>.eurex
rates.opt.<symbol_family>.<metric>.eurex
risk.index.<name>.<metric>.eurex
```

Examples:
- `rates.fut.bund_fut.volume.eurex`
- `rates.fut.bund_fut.open_interest.eurex`
- `rates.opt.bund_opt.volume.eurex`
- `rates.opt.bund_opt.open_interest.eurex`

Where `<symbol_family>` is a stable family key, not per-expiry.

### 7.5 `eurex.refdata.contracts`

Reference table uses contract-level entity IDs:

Template:
```
rates.<instrument>.<symbol>.<yyyymm>.eurex
```

Example:
- `rates.fut.fgbm.202606.eurex`  (if `fgbm` is a contract code)

Contract-level entities are for mastering and mapping; downstream stats should map to family.

### 7.6 `lch.cdsclear.daily`

Template:
```
credit.cds.<segment>.<metric>.lch
```

Examples:
- `credit.cds.itraxx.notional_sum.lch`
- `credit.cds.cdx.notional_sum.lch`
- `credit.cds.all.trade_count.lch`

### 7.7 `cme.productslate.reference`

Reference entities:

Template:
```
<domain>.<instrument>.<product_code>.cme
```

Examples:
- `rates.fut.zq.cme`  (if product_code is ZQ)
- `rates.opt.ozq.cme` (example only)

This table is for mapping; do not treat as time series.

### 7.8 `ezoic.adrevenue.daily`

Template:
```
macro.index.adrevenue.<geo_or_all>.value.ezoic
```

Default:
- `macro.index.adrevenue.global.value.ezoic`

If splits exist:
- `macro.index.adrevenue.us.value.ezoic`
- `macro.index.adrevenue.uk.value.ezoic`

---

## 8. Helper functions (must implement in `public_web/utils.py`)

### 8.1 `make_entity_id(*parts: str) -> str`

- lowercases
- replaces spaces with `_`
- strips invalid chars
- joins with `.`

### 8.2 `normalize_ccy(s: str) -> str|None`

- maps `EURO` -> `eur`, `YEN` -> `jpy`, etc.

### 8.3 `normalize_pair(base: str, quote: str) -> str`

- returns `base+quote` lowercase, with sanity checks

### 8.4 `bucket_tenor(raw) -> str`

- accepts:
  - strings like "2Y", "6M", "1W"
  - dates (effective/maturity) -> approximate tenor
  - floats (years)
- returns standard bucket or `unk`

### 8.5 `eurex_family_key(contract_symbol: str, product_name: str, underlying: str|None) -> str`

- maps contract-level identifiers to stable families:
  - `FGBL` -> `bund_fut`
  - `FGBM` -> `bob1_fut` (example)
- stored in a mapping file for transparency.

---

## 9. Backward compatibility policy

Once introduced, entity ID formats MUST remain stable.
If a schema change is required:
- add a new table name or a version suffix in the table, not in entity IDs.

---

## 10. Quick examples (sanity check)

DTCC PPD daily counts for EUR OIS 2y:
- `table = dtcc.ppd.daily`
- `entity_id = rates.ois.eur.2y.dtccppd`
- metrics are columns: `trade_count`, `notional_sum`, `price_std`, etc.

Eurex Bund futures OI:
- `table = eurex.stats.daily`
- `entity_id = rates.fut.bund_fut.open_interest.eurex`
- `open_interest` column (or use `value` if you decide single-column table; in this pack, it is a dedicated column)

Ezoic ad revenue:
- `table = ezoic.adrevenue.daily`
- `entity_id = macro.index.adrevenue.global.value.ezoic`
- `value` column
