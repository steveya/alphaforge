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

`upsert_pit_observations(..., strict=...)` supports ingestion policy modes:

- `strict="error"` (or `True`): enforce PIT validation before writes.
- `strict="warn"` (or `False`): continue write and emit `PITValidationWarning`.
- `strict="coerce"`: repair/drop irrecoverable rows deterministically before write.

Error mode rejects:

- missing required columns
- nulls in required fields
- duplicate PIT keys in the input frame
- invalid timestamps/timezone issues in release/asof columns
- future rows (`obs_date > asof_utc`)

`validate_pit_observations(df)` returns a `PITValidationReport` for preflight checks.

## Release helper contract

Release stream helpers for reference periods:

- `list_release_stream(series_key, ref, asof=None, freq=None)`
  - returns one ref-period stream ordered by `asof_utc` with `release_rank`, `is_first`, and `is_latest`.
- `resolve_release(series_key, ref, policy=..., asof=None, freq=None)`
  - supports policies: `"first"`, `"latest"`, `{"mode":"rank","rank":n}`, `{"mode":"horizon","horizon":...}`.

## Expression graph contract

Expression graphs define deterministic, dependency-ordered multi-series PIT transforms.

- `explain_expression_graph(...)`
- `preview_expression_graph(...)`
- `apply_expression_graph(...)`

Expression grammar v1:

- operators: `+`, `-`, `*`, `/`, parentheses
- function calls: `lag(alias, n)`, `diff(alias, n)`
- no arbitrary callable execution

Each node applies deterministic as-of alignment using union vintages of direct inputs.

## Vintage union and snapshot panel contract

- `list_union_vintages(series_keys, start, end, mode=\"event|calendar\")`
- `build_snapshot_panel(series_specs, asof, align=\"month_end|quarter_end\", join=...)`

Snapshot panels support per-series release policies and deterministic ref alignment.

## PIT contract versioning

Version API:

- `PIT_CONTRACT_VERSION`
- `get_pit_contract_version()`

Migration entries for contract/validation changes are recorded in `docs/guides/pit-migrations.md`.

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


## Type annotations

All public PIT APIs carry complete type annotations. `release_policy` parameters accept `ReleaseSelectionPolicy | Mapping[str, Any] | str`; the TypedDict union is narrowed internally before key access.
