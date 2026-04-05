# Temporal Semantics

Alphaforge's temporal core distinguishes four different concepts that are easy
to blur together in messy real-world data:

| Concept | Meaning | Typical field |
| --- | --- | --- |
| Observation time | When the measured phenomenon happened | `obs_date` |
| Reference period | The labeled period the value summarizes | `RefPeriod` |
| Release date | When a source is expected or observed to publish the value | `release_time_utc`, `ReleaseRule` |
| Availability / as-of | The information set cutoff used for evaluation | `asof_utc` |

These are related, but they are not interchangeable. A March observation can
belong to the `2025Q1` reference period, be released in May, and still be
unavailable for an April `asof`.

## Reference periods

The canonical typed surface for reference periods is
[`alphaforge.time.ref_period`](../api/time-ref-period.md):

```python
from alphaforge.time import RefFreq, RefPeriod, coerce_ref_period
```

Use `RefPeriod` when the semantic object is a labeled period rather than a raw
timestamp:

- `2025Q1`
- `2025-02`
- `2025`

The library now standardizes three related operations:

- parsing canonical ref keys such as `2025Q1`
- normalizing pandas periods and explicit observation dates through one typed path
- resolving the observation-date anchor explicitly as `"start"` or `"end"`

```python
import pandas as pd
from alphaforge.time import RefFreq, RefPeriod

quarter = RefPeriod.parse("2025Q1")
quarter.to_key()                 # "2025Q1"
quarter.obs_date(anchor="end")   # 2025-03-31 UTC
quarter.obs_date(anchor="start") # 2025-01-01 UTC

RefPeriod.parse(pd.Period("2025Q1", freq="Q"))
RefPeriod.parse("2025-03-31", freq=RefFreq.Q, obs_date_anchor="end")
RefPeriod.parse("2025-01-01", freq=RefFreq.Q, obs_date_anchor="start")
```

That keeps the normalization rule explicit: if you are converting an
observation date into a reference period, you must also say which frequency and
anchor semantics you mean.

## Canonical release-rule API

Release schedules now live in [`alphaforge.time`](../api/pit-release-rules.md),
not just under the PIT package:

```python
from alphaforge.time import FixedLagMonths, QuarterlyRelease, ReleaseRule
```

Use release rules for expected-publication semantics:

- catalog metadata
- release-aware missingness classification
- source health policies when cadence alone is too coarse

For weekly schedules, `WeeklyRelease` now combines:

- the configured lag anchor (`lag_days`), and
- the configured publication weekday (`release_weekday`)

The expected release date is the first configured weekday on or after the lag
anchor, rather than a raw day offset that ignores the weekday field.

Release rules are expectation objects, not a replacement for realized PIT
timestamps. If source data gives an observed `release_time_utc` or `asof_utc`,
that realized timing wins.

## Missingness taxonomy

The canonical missingness classifier also lives in
[`alphaforge.time`](../api/pit-missingness.md):

```python
from alphaforge.time import MissingnessReason, classify_missingness
```

The current taxonomy is:

- `STRUCTURAL`: the panel grid asks for a value that should not exist there
  (for example, a quarterly series on a non-quarter-end monthly slot)
- `FUTURE`: the observation period has not ended yet
- `RAGGED_EDGE`: the value is not expected to have been released yet
- `TRUE_MISSING`: the value should have been available but is absent

This keeps PIT panel logic, release-aware reasoning, and downstream data
quality checks on the same vocabulary.

## Health semantics

`alphaforge.pipeline.health.SourceHealthPolicy` can now use a `release_rule` to
evaluate staleness against the next expected release date instead of only the
time since the latest observation date.

This matters for sources where:

- observation cadence and publication cadence differ materially
- long release lags would make cadence-only health checks look stale too early
- monthly or quarterly data should stay `ok` until the next scheduled release
  window actually passes

## Compatibility

The preferred imports are now:

- `alphaforge.time.release_rules`
- `alphaforge.time.missingness`
- `alphaforge.time.ref_period`
- `from alphaforge.time import ...`

Compatibility shims remain in place for:

- `alphaforge.pit.release_rules`
- `alphaforge.pit.missingness`

Those PIT paths still work, but they are now legacy entry points rather than
the canonical temporal-semantics surface.
