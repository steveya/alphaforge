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

`PITAccessor.upsert_pit_observations(df, strict=...)` supports three policy modes:

- `strict=\"error\"` (or `True`): block writes on validation errors.
- `strict=\"warn\"` (or `False`): continue writes and emit `PITValidationWarning`.
- `strict=\"coerce\"`: normalize and repair rows (drop irrecoverable rows deterministically).

Strict mode rejects:

- missing required columns
- nulls in required fields
- duplicate PIT keys in the input frame
- timezone issues in `asof_utc` / `release_time_utc`
- future rows where `obs_date > asof_utc`

## Release/ref helpers

```python
stream = pit.list_release_stream(\"GDP\", \"2024Q4\", asof=pd.Timestamp(\"2025-03-31\", tz=\"UTC\"))
record = pit.resolve_release(
    \"GDP\",
    \"2024Q4\",
    policy={\"mode\": \"rank\", \"rank\": 2},
    asof=pd.Timestamp(\"2025-03-31\", tz=\"UTC\"),
)
```

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

### Multi-step pipelines (preview + incremental apply)

Pipelines compose multiple PIT transforms into an ordered DAG while preserving PIT semantics.

```python
from alphaforge.pit.pipelines import PITPipelineSpec, PITPipelineStep

pipeline = PITPipelineSpec(
    pipeline_id="macro/gdp_pipeline",
    steps=(
        PITPipelineStep(
            name="lag",
            spec={
                "input_series_key": "GDP",
                "output_series_key": "GDP_lag1",
                "op": "lag",
                "params": {"periods": 1},
            },
        ),
        PITPipelineStep(
            name="diff",
            spec={
                "input_series_key": "GDP_lag1",
                "output_series_key": "GDP_lag1_diff",
                "op": "diff",
                "params": {"periods": 1},
            },
            depends_on=("lag",),
        ),
    ),
)

plan = pit.explain_pipeline(pipeline, incremental=True)
preview = pit.preview_pipeline(pipeline, overwrite=True)
result = pit.apply_pipeline(pipeline, overwrite=True, incremental=True)
runs = pit.list_pipeline_runs(result.pipeline_id, limit=5)
```

## Expression graph API

```python
from alphaforge.pit.models import PITExpressionGraphSpec, PITExpressionNode

graph = PITExpressionGraphSpec(
    graph_id=\"macro/bridge_demo\",
    nodes=(
        PITExpressionNode(
            name=\"spread\",
            output_series_key=\"GDP_minus_CPI_expr\",
            expression=\"gdp - lag(cpi, 1)\",
            inputs={\"gdp\": \"GDP\", \"cpi\": \"CPI\"},
            join=\"inner\",
        ),
    ),
)

plan = pit.explain_expression_graph(graph)
preview = pit.preview_expression_graph(graph, overwrite=True)
result = pit.apply_expression_graph(graph, overwrite=True)
```

## Union vintages and snapshot panels

```python
vintages = pit.list_union_vintages([\"GDP\", \"CPI\"], mode=\"event\")
panel = pit.build_snapshot_panel(
    [
        {\"series_key\": \"GDP\", \"alias\": \"gdp\"},
        {\"series_key\": \"CPI\", \"alias\": \"cpi\", \"release_policy\": \"latest\"},
    ],
    asof=pd.Timestamp(\"2025-06-30\", tz=\"UTC\"),
    align=\"month_end\",
    join=\"outer\",
)
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

```python
# docs-example:pit_pipeline_incremental
from pathlib import Path
import pandas as pd

from alphaforge.pit import PITAccessor
from alphaforge.pit.pipelines import PITPipelineSpec, PITPipelineStep
from alphaforge.pit.transforms import PITTransformSpec
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

TMP_DIR = globals().get("TMP_DIR", Path("./tmp"))
store = DuckDBParquetStore(root=str(TMP_DIR / "pit_docs_example_4"))
pit = PITAccessor(store.conn())

pit.upsert_pit_observations(
    pd.DataFrame(
        {
            "series_key": ["GDP", "GDP", "GDP"],
            "obs_date": [
                pd.Timestamp("2024-12-31"),
                pd.Timestamp("2025-03-31"),
                pd.Timestamp("2025-06-30"),
            ],
            "asof_utc": [
                pd.Timestamp("2025-01-10", tz="UTC"),
                pd.Timestamp("2025-04-10", tz="UTC"),
                pd.Timestamp("2025-07-10", tz="UTC"),
            ],
            "value": [1.0, 2.0, 3.0],
        }
    )
)

pipeline = PITPipelineSpec(
    pipeline_id="docs/gdp_pipeline",
    steps=(
        PITPipelineStep(
            name="lag",
            spec=PITTransformSpec(
                input_series_key="GDP",
                output_series_key="GDP_lag1",
                op="lag",
                params={"periods": 1},
            ),
        ),
        PITPipelineStep(
            name="diff",
            spec=PITTransformSpec(
                input_series_key="GDP_lag1",
                output_series_key="GDP_lag1_diff",
                op="diff",
                params={"periods": 1},
            ),
            depends_on=("lag",),
        ),
    ),
)

preview = pit.preview_pipeline(pipeline, overwrite=True)
assert not preview.empty
first_run = pit.apply_pipeline(pipeline, overwrite=True)
assert first_run.status == "success"

pit.upsert_pit_observations(
    pd.DataFrame(
        {
            "series_key": ["GDP"],
            "obs_date": [pd.Timestamp("2025-09-30")],
            "asof_utc": [pd.Timestamp("2025-10-10", tz="UTC")],
            "value": [4.0],
        }
    )
)
second_run = pit.apply_pipeline(pipeline, incremental=True)
assert second_run.status == "success"
assert second_run.effective_start_asof is not None
```
