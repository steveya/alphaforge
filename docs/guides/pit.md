# Point-in-Time (PIT) Data

Alphaforge PIT provides:

1. PIT ingestion and retrieval (`PITAccessor`)
2. PIT-preserving transforms (`apply_transform`, `preview_transform`)
3. PIT safety and integration (`ReleaseLagPolicy`, `PITDataSource`)

## Canonical PIT schema

| Column | Type | Notes |
| --- | --- | --- |
| `series_key` | TEXT | Series identifier |
| `obs_date` | TIMESTAMP | Observation date / period-end |
| `asof_utc` | TIMESTAMP | Vintage (knowledge timestamp) |
| `value` | DOUBLE | Observation value |
| `release_time_utc` | TIMESTAMP | Optional release timestamp |
| `revision_id` | TEXT | Optional revision label |
| `source` | TEXT | Optional source descriptor |
| `meta_json` | TEXT | Optional lineage payload |
| `ingested_utc` | TIMESTAMP | Insert timestamp |

Uniqueness is enforced on `(series_key, obs_date, asof_utc)`.

## Ingestion contract

`PITAccessor.upsert_pit_observations(df, strict=True)` validates PIT rows before write.

In strict mode, ingestion fails on:

- missing required columns
- nulls in required fields
- duplicate PIT keys in the input frame
- timezone issues in `asof_utc` / `release_time_utc`
- future rows where `obs_date > asof_utc`

```python
import pandas as pd
from alphaforge.pit import PITAccessor
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
    ),
    strict=True,
)
```

## Snapshot and timeline retrieval

```python
snapshot = pit.get_snapshot("GDP", pd.Timestamp("2025-02-15", tz="UTC"))
timeline = pit.get_revision_timeline("GDP", pd.Timestamp("2024-12-31", tz="UTC"))
```

## Transform API (`obs_path`)

Use `preview_transform` to inspect rows before persistence, then `apply_transform` to write.

```python
from alphaforge.pit.transforms import PITTransformSpec

spec = PITTransformSpec(
    input_series_key="GDP",
    output_series_key="GDP_q_last",
    axis="obs_path",
    op="resample",
    params={"rule": "Q", "agg": "last"},
    engine="auto",
)

preview = pit.preview_transform(spec)
result = pit.apply_transform(spec, overwrite=True)
print(result.engine_requested, result.engine_used, result.fallback_reason)
```

## Experimental `revision_path`

`revision_path` requires explicit opt-in per call.

```python
spec = PITTransformSpec(
    input_series_key="GDP",
    output_series_key="GDP_revision_delta",
    axis="revision_path",
    op="diff",
    params={"periods": 1},
)

pit.apply_transform(spec, overwrite=True, allow_experimental=True)
```

Without `allow_experimental=True`, Alphaforge raises `PITExperimentalFeatureError`.

## Engine contract

- `engine="auto"` prefers `duckdb` for supported op+axis+params combinations.
- `engine="duckdb"` raises `PITEngineError` when a spec is unsupported.
- You can force fallback with `on_engine_mismatch="fallback"` to run on `python`.
- `path_apply` is Python-only in v1.

```python
pit.apply_transform(
    PITTransformSpec(
        input_series_key="GDP",
        output_series_key="GDP_diff",
        op="diff",
        params={"periods": 1},
        engine="duckdb",
    ),
    on_engine_mismatch="fallback",
)
```

## PITDataSource integration

`PITDataSource` exposes two tables:

- `pit.snapshot` (requires `Query.asof`, supports only `vintage="latest"`)
- `pit.observations` (raw rows with optional as-of filtering, currently `vintage="latest"` only)

```python
from alphaforge.data.context import DataContext
from alphaforge.data.pit_source import PITDataSource
from alphaforge.data.query import Query

pit_source = PITDataSource(pit=pit)
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

## Reference-period helpers

```python
timeline = pit.get_revision_timeline_ref("GDP", "2024Q4")
snapshot = pit.get_snapshot_ref(
    "GDP",
    asof=pd.Timestamp("2025-06-01", tz="UTC"),
    start_ref="2019Q1",
    end_ref="2024Q4",
)
```
