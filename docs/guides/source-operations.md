# Source Operations

This guide covers the shared operational surfaces for source health monitoring
and recurring archive-backed ingestion.

## Release-aware health reports

Use `SourceHealthPolicy` and `assess_source_health(...)` when you need a
source-level health decision at a specific `asof`.

When a source has a typed `release_rule`, Alphaforge now exposes structured
release-aware diagnostics:

- `expected_next`
- `overdue`
- `overdue_days`
- `status`
- `weight_factor`

Turn a set of health statuses into a report frame with
`build_health_report(...)`:

```python
from alphaforge.pipeline.health import (
    SourceHealthPolicy,
    assess_source_health,
    build_health_report,
)
from alphaforge.time import FixedLagMonths

policy = SourceHealthPolicy(
    expected_cadence=pd.Timedelta(days=31),
    release_rule=FixedLagMonths(months=2),
)

status = assess_source_health(
    "macro",
    latest_obs_date=pd.Timestamp("2025-01-31", tz="UTC"),
    asof=pd.Timestamp("2025-04-05", tz="UTC"),
    policy=policy,
)

report = build_health_report({"macro": status})
```

If you are already recording health through `SourceHealthTracker`, use
`tracker.report(asof)` to get the same dataframe surface from the configured
tracker.

## Archive-backed ingestion planning

Recurring public-web ingestion flows often need more than raw URLs. They need a
deterministic fetch plan with artifact names that can be reused in cache and
debug logs.

`alphaforge.data.public_web.archive` now exposes planned fetch entries via
`ArchiveFetchPlanEntry` and helper builders:

- `discover_archive_fetches(...)`
- `iter_yearly_archive_fetches(...)`

These helpers:

- keep deterministic `artifact_name` values
- preserve the resolved fetch `url`
- infer a `year` when it is present in the archive path
- handle query-string download links during archive discovery

Example:

```python
from alphaforge.data.public_web.archive import discover_archive_fetches

planned = discover_archive_fetches(
    html,
    base_url="https://example.com/archive/index.html",
    suffixes=[".zip", ".csv"],
    years=[2024, 2025],
    fallback_artifact_prefix="cftc_swaps_weekly",
)

for entry in planned:
    print(entry.url, entry.artifact_name, entry.year)
```

The archive-backed public-web loaders use this plan layer so recurring
ingestion flows and cache artifact names stay deterministic across runs.

For CFTC archive-backed loaders specifically, broken archive downloads or ZIP
parse failures now fail fast instead of being silently skipped. That keeps
historical gaps observable and prevents partial year ranges from looking like a
successful empty or truncated fetch.
