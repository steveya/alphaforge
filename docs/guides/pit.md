# Point-in-Time (PIT) Data

Alphaforge supports revised macro series using a canonical PIT table and three layers:

1. Snapshot and revision-timeline retrieval (`PITAccessor`)
2. PIT-preserving transforms (`apply_transform`)
3. PIT safety and integration (`ReleaseLagPolicy`, `PITDataSource`)

## Canonical PIT schema

The PIT table is created automatically when using `DuckDBParquetStore`.

| Column | Type | Notes |
| --- | --- | --- |
| series_key | TEXT | Series identifier (for example `"GDP"`) |
| obs_date | TIMESTAMP | Reference period end date |
| asof_utc | TIMESTAMP | Vintage / knowledge time |
| value | DOUBLE | Observed value |
| release_time_utc | TIMESTAMP | Optional release timestamp |
| revision_id | TEXT | Optional revision label |
| source | TEXT | Optional source |
| meta_json | TEXT | Optional JSON-encoded provenance |
| ingested_utc | TIMESTAMP | Default `now()` at insert |

Uniqueness is enforced on `(series_key, obs_date, asof_utc)`.

## Timezone handling

Input timestamps are normalized to timezone-aware UTC on ingestion. Snapshot and revision queries return UTC indexes.

## Core usage

```python
import pandas as pd
from alphaforge.pit.accessor import PITAccessor
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

store = DuckDBParquetStore(root="./store")
pit = PITAccessor(store.conn())

pit.upsert_pit_observations(
    pd.DataFrame(
        {
            "series_key": ["GDP", "GDP"],
            "obs_date": [pd.Timestamp("2024-12-31"), pd.Timestamp("2024-12-31")],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-02-10", tz="UTC"),
            ],
            "value": [1.0, 1.1],
        }
    )
)

snapshot = pit.get_snapshot("GDP", pd.Timestamp("2025-02-15", tz="UTC"))
timeline = pit.get_revision_timeline("GDP", pd.Timestamp("2024-12-31", tz="UTC"))
```

## Transform API (obs_path)

`PITAccessor.apply_transform` creates derived PIT series while preserving as-of causality.

```python
from alphaforge.pit.transforms import PITTransformSpec

spec = PITTransformSpec(
    input_series_key="GDP",
    output_series_key="GDP_q_last",
    axis="obs_path",
    op="resample",
    params={"rule": "Q", "agg": "last"},
)

result = pit.apply_transform(spec, overwrite=True)
print(result.transform_id, result.rows_written)
```

Supported `obs_path` operators:

- `resample`, `aggregate`, `rolling`, `expanding`, `lag`, `diff`, `path_apply`

## Experimental transform API (`revision_path`)

`revision_path` is experimental in this release and currently supports:

- `diff`, `lag`, `rolling`, `expanding`

```python
spec = PITTransformSpec(
    input_series_key="GDP",
    output_series_key="GDP_revision_delta",
    axis="revision_path",
    op="diff",
    params={"periods": 1},
)
pit.apply_transform(spec, overwrite=True)
```

For `revision_path`, transforms run along each `(series_key, obs_date)` revision timeline ordered by `asof_utc`.

## Lag policy and leakage guards

```python
from alphaforge.pit.guards import ReleaseLagPolicy, effective_asof, pit_leakage_report

policy = ReleaseLagPolicy(default_lag=pd.Timedelta(days=30))
cutoff = effective_asof(pd.Timestamp("2025-04-15", tz="UTC"), "GDP", policy)

report = pit_leakage_report(
    pd.DataFrame(
        {
            "series_key": ["GDP"],
            "obs_date": [pd.Timestamp("2025-03-31", tz="UTC")],
            "asof_utc": [pd.Timestamp("2025-04-15", tz="UTC")],
            "value": [2.1],
        }
    )
)
```

## Dataset integration via PITDataSource

```python
from alphaforge.data.context import DataContext
from alphaforge.data.pit_source import PITDataSource
from alphaforge.data.query import Query

pit_source = PITDataSource(pit=pit, lag_policy=policy)
ctx = DataContext(sources={"pit": pit_source}, calendars={}, store=store)

panel = ctx.fetch_panel(
    "pit",
    Query(
        table="pit.snapshot",
        columns=["value", "asof_utc"],
        entities=["GDP_q_last"],
        asof=pd.Timestamp("2025-06-01", tz="UTC"),
    ),
)
```

Tables exposed by `PITDataSource`:

- `pit.snapshot`: latest-leq snapshot view (`Query.asof` required)
- `pit.observations`: raw vintage rows with optional as-of filtering

## Reference period keys

PIT ref helpers map keys to `obs_date` end timestamps (UTC midnight).

Supported formats:

- Annual: `YYYY`
- Quarterly: `YYYYQq`
- Monthly: `YYYY-MM` or `YYYY/MM`
- Month-end date: `YYYY-MM-DD` (interpreted as monthly)

Example:

```python
timeline = pit.get_revision_timeline_ref("GDP", "2024Q4")
snapshot = pit.get_snapshot_ref(
    "GDP",
    asof=pd.Timestamp("2025-06-01", tz="UTC"),
    start_ref="2019Q1",
    end_ref="2024Q4",
)
```
