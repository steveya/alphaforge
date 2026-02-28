# Point-in-Time (PIT) Data

Alphaforge PIT provides:

1. PIT ingestion and retrieval (`PITAccessor`)
2. PIT-preserving transforms (`preview_transform`, `apply_transform`)
3. PIT safety and integration (`ReleaseLagPolicy`, `PITDataSource`)
4. Revision/staleness helpers (`alphaforge.pit.tasks`)

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

Strict mode rejects:

- missing required columns
- nulls in required fields
- duplicate PIT keys in the input frame
- timezone issues in `asof_utc` / `release_time_utc`
- future rows where `obs_date > asof_utc`

## Transform API

### Single-series path transforms (`obs_path`)

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
```

### Cross-series transform (`op="binary"`)

Cross-series transforms are PIT-safe and still produce PIT output.

```python
spec = PITTransformSpec(
    input_series_key="GDP",
    output_series_key="GDP_minus_CPI",
    axis="obs_path",
    op="binary",
    params={
        "right_series_key": "CPI",
        "operator": "sub",  # add | sub | mul | div
        "join": "inner",    # inner | left | right | outer
    },
    engine="auto",
)

result = pit.apply_transform(spec, overwrite=True)
```

### Experimental revision timeline transforms (`revision_path`)

`revision_path` requires explicit opt-in:

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

## Engine contract

- `engine="auto"` prefers `duckdb` for supported op+axis+params combinations.
- `engine="duckdb"` raises `PITEngineError` when a spec is unsupported.
- Set `on_engine_mismatch="fallback"` to force deterministic fallback to `python`.
- `path_apply` and `binary` currently execute on Python only.

## Revision analytics and staleness helpers

Available in `alphaforge.pit.tasks`:

- vintage selectors (`first_vintage_snapshot`, `latest_vintage_snapshot`, `snapshot_at_horizon`)
- revision helpers (`revision_deltas`, `revision_events`, `revision_event_stream`, `revision_stability`, `revision_volatility`)
- staleness helper (`forward_fill_with_staleness`)
- growth helpers (`yoy`, `qoq`)

`forward_fill_with_staleness` returns:

- `value`
- `source_obs_date`
- `age` (`Timedelta`)
- `is_stale`
- `age_days`

## PITDataSource integration

`PITDataSource` exposes two tables:

- `pit.snapshot`: requires `Query.asof`, currently supports only `vintage="latest"`
- `pit.observations`: raw rows with optional as-of filtering, currently supports only `vintage="latest"`

## Anti-leakage checklist

1. Always specify `Query.asof` for `pit.snapshot` queries.
2. Use `ReleaseLagPolicy` for delayed-release indicators.
3. Prefer `preview_transform` before writing transformed series.
4. Keep `strict=True` ingestion for raw observations.
5. For cross-series transforms, verify both inputs exist and are aligned for expected joins.
6. Run `pit_leakage_report` on intermediate frames used outside PIT storage.

## Runnable examples (CI-verified)

```python
# docs-example:pit_ingest_snapshot
from pathlib import Path
import pandas as pd

from alphaforge.pit import PITAccessor
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

TMP_DIR = globals().get("TMP_DIR", Path("./tmp"))
store = DuckDBParquetStore(root=str(TMP_DIR / "pit_docs_example_1"))
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

snapshot = pit.get_snapshot("GDP", pd.Timestamp("2025-02-15", tz="UTC"))
assert not snapshot.empty
```

```python
# docs-example:pit_cross_series_binary
from pathlib import Path
import pandas as pd

from alphaforge.pit import PITAccessor
from alphaforge.pit.transforms import PITTransformSpec
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

TMP_DIR = globals().get("TMP_DIR", Path("./tmp"))
store = DuckDBParquetStore(root=str(TMP_DIR / "pit_docs_example_2"))
pit = PITAccessor(store.conn())

pit.upsert_pit_observations(
    pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "CPI", "CPI"],
            "obs_date": [
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
                pd.Timestamp("2024-01-31"),
                pd.Timestamp("2024-02-29"),
            ],
            "asof_utc": [
                pd.Timestamp("2024-03-15", tz="UTC"),
                pd.Timestamp("2024-03-15", tz="UTC"),
                pd.Timestamp("2024-03-15", tz="UTC"),
                pd.Timestamp("2024-03-15", tz="UTC"),
            ],
            "value": [3.0, 3.3, 1.0, 1.2],
        }
    )
)

spec = PITTransformSpec(
    input_series_key="GDP",
    output_series_key="GDP_minus_CPI",
    op="binary",
    params={"right_series_key": "CPI", "operator": "sub", "join": "inner"},
)
result = pit.apply_transform(spec, overwrite=True)
assert result.rows_written > 0
```

```python
# docs-example:pit_revision_and_staleness_tasks
from pathlib import Path
import pandas as pd

from alphaforge.pit import PITAccessor
from alphaforge.pit.tasks import (
    forward_fill_with_staleness,
    revision_event_stream,
    revision_volatility,
)
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

TMP_DIR = globals().get("TMP_DIR", Path("./tmp"))
store = DuckDBParquetStore(root=str(TMP_DIR / "pit_docs_example_3"))
pit = PITAccessor(store.conn())

pit.upsert_pit_observations(
    pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "GDP"],
            "obs_date": [
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2025-03-31"),
            ],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-02-10", tz="UTC"),
                pd.Timestamp("2025-04-10", tz="UTC"),
            ],
            "value": [1.0, 1.2, 2.0],
        }
    )
)

events = revision_event_stream(pit, "GDP", min_abs_change=0.05)
vol = revision_volatility(pit, "GDP")
assert not events.empty
assert not vol.empty

snapshot = pit.get_snapshot("GDP", pd.Timestamp("2025-04-15", tz="UTC"))
filled = forward_fill_with_staleness(
    snapshot,
    max_staleness=pd.Timedelta(days=60),
    target_index=pd.date_range("2024-12-31", periods=4, freq="ME", tz="UTC"),
)
assert {"value", "source_obs_date", "age", "is_stale", "age_days"}.issubset(filled.columns)
```
