# PIT API Contract

This guide defines stable contracts for PIT ingestion, transforms, and data-source queries.

## Error model

PIT uses typed exceptions:

- `PITError`
- `PITContractError`
- `PITValidationError`
- `PITUnsupportedOperationError`
- `PITExperimentalFeatureError`
- `PITCausalityError`
- `PITEngineError`

Use these for deterministic handling in client code instead of parsing generic error strings.

## Transform spec contract

`PITTransformSpec` is the canonical transform input. `PITAccessor` also accepts a mapping with equivalent fields.

Allowed operators by axis:

- `obs_path`: `resample`, `aggregate`, `rolling`, `expanding`, `lag`, `diff`, `binary`, `path_apply`
- `revision_path` (experimental): `rolling`, `expanding`, `lag`, `diff`

Unknown parameter keys are rejected per operator.

`binary` operator contract (`obs_path` only):

- `right_series_key` (required)
- `operator` in `add | sub | mul | div`
- `join` in `inner | left | right | outer` (default `inner`)
- optional `fill_value`

## Pipeline contract

`PITPipelineSpec` defines a named collection of `PITPipelineStep` nodes with optional dependencies.

- each step has unique `name`
- `depends_on` must reference existing step names
- execution order is deterministic and dependency-safe
- step transforms use the same validation/engine/experimental contracts as `apply_transform`

Pipeline APIs:

- `PITAccessor.explain_pipeline(...)`
- `PITAccessor.preview_pipeline(...)`
- `PITAccessor.apply_pipeline(...)`
- `PITAccessor.list_pipelines(...)`
- `PITAccessor.list_pipeline_runs(...)`

Incremental controls:

- `incremental=True` enables anchored execution
- `since_asof` sets an explicit as-of anchor
- `since_run_id` anchors to a prior pipeline run
- if no explicit anchor is provided, incremental runs anchor to the previous successful run’s max output as-of

## Engine contract

PIT transforms support two execution backends:

- `duckdb`: built-ins (`resample`, `aggregate`, `rolling`, `expanding`, `lag`, `diff`) with supported parameters
- `python`: full v1 operator coverage (including `path_apply` and `binary`)

- `engine="auto"` -> uses `duckdb` for supported specs, otherwise `python`
- `engine="python"` -> uses `python`
- `engine="duckdb"`:
  - `on_engine_mismatch="error"` -> raises `PITEngineError`
  - `on_engine_mismatch="fallback"` -> uses `python` and records `fallback_reason`

`PITTransformResult` reports both requested and effective engine.

## Experimental gating

`axis="revision_path"` requires explicit opt-in:

- `allow_experimental=False` (default) -> raises `PITExperimentalFeatureError`
- `allow_experimental=True` -> executes allowed revision-path ops

## Ingestion validation contract

`upsert_pit_observations(..., strict=True)` enforces PIT validation before writes.

Strict mode rejects:

- missing required columns
- nulls in required fields
- duplicate PIT keys in the input frame
- invalid timestamps/timezone issues in release/asof columns
- future rows (`obs_date > asof_utc`)

`validate_pit_observations(df)` returns a `PITValidationReport` for preflight checks.

## Data-source query contract

`PITDataSource` table semantics:

- `pit.snapshot`
  - requires `Query.asof`
  - supports only `Query.vintage == "latest"`
  - rejects `Query.vintage_id`
- `pit.observations`
  - supports only `Query.vintage == "latest"`
  - rejects `Query.vintage_id`

Use `PITDataSource.snapshot_query(...)` and `PITDataSource.observations_query(...)` helper constructors for safe defaults.
