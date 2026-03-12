# Point-in-Time (PIT) Data

Alphaforge PIT provides:

1. PIT ingestion and retrieval (`PITAccessor`)
2. PIT-preserving transforms (`preview_transform`, `apply_transform`)
3. PIT execution contexts for folds and sequence/tape materialization
4. PIT safety and integration (`ReleaseLagPolicy`, `PITDataSource`)
5. Revision/staleness helpers (`alphaforge.pit.tasks`)
6. Model-importance attribution back to dataset source fields and request tags

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

### PIT operator capability map

| Capability | Status | Notes |
| --- | --- | --- |
| Single-series temporal ops | supported now | `lag`, `diff`, `pct_change`, `ffill` |
| Temporal aggregation | supported now | `resample`, `aggregate`, `rolling`, `expanding` with `first`, `last`, `min`, `max`, `mean`, `sum`, `count`, `std`, `var` |
| PIT-safe cross-series arithmetic | supported now | `binary` |
| PIT-safe ordered fallback | supported now | `coalesce` with ordered precedence and row-level selected-input lineage |
| PIT-safe adjusted continuation | supported now | `splice` with `ratio` / `add`, hard switch or transition blending, and row-level calibration lineage |
| Multi-series formulas | supported via expression graph | deterministic alias-based graphs, union-vintage aligned |
| Walk-forward and purged PIT folds | supported now | explicit as-of-grid based fold generators |
| Live-safe snapshot tapes | supported now | `filtered` mode uses each step’s own as-of |
| Retrospective research tapes | research-only | `smoothed_research` requires explicit opt-in |
| Broader SQL/pandas parity | planned next | more operators will be layered on the same PIT operator framework instead of a separate DSL |

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

Additional `obs_path` operators:

- `pct_change` with `params={"periods": n}`
- `ffill` with optional `params={"limit": n}`
- `coalesce` with `params={"other_series_keys": [...]}` for ordered PIT-safe fallback
- `splice` with `params={"right_series_key": ..., "adjustment": "ratio|add", "transition_periods": n, "join": "outer|inner|left|right"}` for PIT-safe adjusted continuation
- `rolling` / `expanding` / `aggregate` / `resample` now also support `count`, `std`, and `var`

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

### PIT-safe splice + attribution example

```python
# docs-example: pit_splice_importance
import importlib.util
from pathlib import Path

example_path = Path.cwd() / "examples" / "pit_splice_importance.py"
spec = importlib.util.spec_from_file_location("pit_splice_importance_example", example_path)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)

outputs = module.run_example(TMP_DIR)
assert outputs["selected_fallback_series_key"] == "ALT"
assert outputs["selected_fallback_asof"] == "2024-04-05T00:00:00+00:00"
assert float(outputs["tag_rollup"].loc["raw", "importance"]) == 0.7
```

This example does four things end-to-end:

1. Ingests revised PIT observations.
2. Splices a sparse primary series with `coalesce`.
3. Derives PIT-safe temporal transforms (`pct_change`, `aggregate(count)`).
4. Builds a dataset artifact, stamps `FeatureRequest.tags` into the catalog, and rolls model importance back up by source fields and tags.

### Adjusted splice (`op="splice"`)

`splice` is pairwise and `obs_path`-only in this milestone.

- `ratio` rescales the right-hand series using the last overlapping non-null point visible inside the same as-of snapshot.
- `add` shifts the right-hand series by the same anchor rule.
- `transition_periods=0` is a hard switch.
- `transition_periods>0` performs a PIT-safe linear blend from left to adjusted-right over the first `n` obs dates at or after the right-hand handoff date.
- If no overlap is visible yet, the adjusted handoff region remains unavailable and is omitted from persisted output until calibration exists.

```python
spec = PITTransformSpec(
    input_series_key="OLD_SERIES",
    output_series_key="CONT_SERIES",
    axis="obs_path",
    op="splice",
    params={
        "right_series_key": "NEW_SERIES",
        "adjustment": "ratio",      # ratio | add
        "transition_periods": 2,    # optional
        "join": "outer",            # optional
    },
)

preview = pit.preview_transform(spec)
result = pit.apply_transform(spec, overwrite=True)
```

Row lineage records:

- both source series keys
- source as-of timestamps by side
- anchor obs date and anchor values
- computed scale/offset
- transition length and per-row blend weights

## PIT execution contexts

### Walk-forward and purged folds

Fold generators operate on an explicit as-of grid instead of inferring business-day spacing.

```python
# docs-example: pit_execution_contexts
import pandas as pd
from alphaforge.pit import (
    PITAccessor,
    PITTapeSpec,
    SnapshotSeriesSpec,
    build_snapshot_tape,
    iter_purged_kfold_folds,
    iter_walk_forward_folds,
)
from alphaforge.store.duckdb_parquet import DuckDBParquetStore

store = DuckDBParquetStore(root=str(TMP_DIR))
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

asof_grid = pd.DatetimeIndex(
    [
        pd.Timestamp("2025-01-15", tz="UTC"),
        pd.Timestamp("2025-02-15", tz="UTC"),
        pd.Timestamp("2025-03-15", tz="UTC"),
        pd.Timestamp("2025-04-15", tz="UTC"),
    ]
)

walk_forward = list(
    iter_walk_forward_folds(
        asof_grid,
        train_size=2,
        validation_size=1,
        purge=1,
    )
)
purged = list(iter_purged_kfold_folds(asof_grid, n_splits=2, purge=1, embargo=0))

filtered_tape = build_snapshot_tape(
    pit,
    PITTapeSpec(
        series_specs=(SnapshotSeriesSpec(series_key="GDP", alias="gdp_latest"),),
        step_asofs=(pd.Timestamp("2025-01-15", tz="UTC"), pd.Timestamp("2025-03-15", tz="UTC")),
        mode="filtered",
    ),
)
smoothed_tape = build_snapshot_tape(
    pit,
    {
        "series_specs": [{"series_key": "GDP", "alias": "gdp_latest"}],
        "step_asofs": [
            pd.Timestamp("2025-01-15", tz="UTC"),
            pd.Timestamp("2025-03-15", tz="UTC"),
        ],
        "mode": "smoothed_research",
    },
    allow_research=True,
)

assert len(walk_forward) == 1
assert len(purged) == 2
filtered_step = filtered_tape[
    (filtered_tape["step_asof_utc"] == pd.Timestamp("2025-01-15", tz="UTC"))
    & (filtered_tape["obs_date"] == pd.Timestamp("2024-12-31", tz="UTC"))
].iloc[0]
smoothed_step = smoothed_tape[
    (smoothed_tape["step_asof_utc"] == pd.Timestamp("2025-01-15", tz="UTC"))
    & (smoothed_tape["obs_date"] == pd.Timestamp("2024-12-31", tz="UTC"))
].iloc[0]
assert filtered_step["value"] == 1.0
assert smoothed_step["value"] == 1.2
assert smoothed_step["sequence_mode"] == "smoothed_research"
```

### Snapshot tape contract

`build_snapshot_tape(...)` returns a long-form frame with:

- `step_asof_utc`
- `materialized_asof_utc`
- `obs_date`
- `series_key`
- `series_alias`
- `value`
- `source_asof_utc`
- `sequence_mode`

Use modes as follows:

- `filtered`: decision-safe, validation-safe, and live-like. Each step uses only data visible at that step’s own as-of.
- `smoothed_research`: explicit retrospective mode for smoothing/backward-pass research. Each step is materialized with a terminal retrospective as-of and requires `allow_research=True`.

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
- `pct_change` is available on both Python and DuckDB.
- `ffill`, `coalesce`, `splice`, `path_apply`, and `binary` currently execute on Python only.

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
