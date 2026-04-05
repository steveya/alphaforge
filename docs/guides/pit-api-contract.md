# PIT API Contract

This guide defines stable contracts for PIT ingestion, transforms, and data-source queries.

The repo-local regression gates for this contract live in
[Core Platform Contracts And Benchmarks](contracts-and-benchmarks.md).

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

- `obs_path`: `resample`, `aggregate`, `rolling`, `expanding`, `lag`, `diff`, `pct_change`, `ffill`, `binary`, `coalesce`, `splice`, `path_apply`
- `revision_path` (experimental): `rolling`, `expanding`, `lag`, `diff`

Unknown parameter keys are rejected per operator.

`binary` operator contract (`obs_path` only):

- `right_series_key` (required)
- `operator` in `add | sub | mul | div`
- `join` in `inner | left | right | outer` (default `inner`)
- optional `fill_value`

`coalesce` operator contract (`obs_path` only):

- `input_series_key` is the highest-precedence source
- `other_series_keys` is required and defines fallback precedence order
- output alignment is on the union of obs dates visible inside each as-of snapshot
- row lineage records `selected_input_series_key` and `selected_input_asof_utc`

`splice` operator contract (`obs_path` only):

- `right_series_key` is required
- `adjustment` is required and must be `ratio` or `add`
- optional `transition_periods >= 0` controls PIT-safe linear blending during handoff
- optional `join` controls obs-date alignment and defaults to `outer`
- calibration uses the last overlapping non-null point visible in the same as-of snapshot
- if no overlap is visible yet, adjusted handoff rows remain unavailable until calibration exists
- row lineage records:
  - left/right source as-of timestamps
  - anchor obs date
  - anchor left/right values
  - computed scale/offset
  - transition weights

`pct_change` / `ffill` contract (`obs_path` only):

- `pct_change` requires `periods > 0` and matches pandas `fill_method=None`
- `ffill` accepts optional `limit > 0`

Aggregation contract:

- `resample`, `aggregate`, `rolling`, and `expanding` support `first`, `last`, `min`, `max`, `mean`, `sum`, `count`, `std`, `var`
- `std` and `var` follow sample statistics semantics (`ddof=1`)

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

## PIT fold contract

PIT fold generators operate on explicit as-of grids and yield `PITFoldSpec` outputs.

Fold output fields:

- `fold_id`
- `fold_mode` in `walk_forward | purged_kfold`
- `train_asofs`
- `validation_asofs`
- `purge`
- `embargo`

Fold APIs:

- `iter_walk_forward_folds(...)`
- `iter_purged_kfold_folds(...)`

Semantics:

- folds are derived from sorted unique as-of timestamps
- purge and embargo are counts on the provided as-of grid, not business-day offsets
- walk-forward folds are decision-safe by construction because training as-ofs are strictly earlier than validation as-ofs

## PIT tape contract

`PITTapeSpec` defines a snapshot-tape materialization request:

- `series_specs`
- `step_asofs`
- `mode` in `filtered | smoothed_research`
- optional `terminal_asof` for retrospective materialization

Tape API:

- `build_snapshot_tape(...)`

Tape output columns:

- `step_asof_utc`
- `materialized_asof_utc`
- `obs_date`
- `series_key`
- `series_alias`
- `value`
- `source_asof_utc`
- `sequence_mode`

Mode semantics:

- `filtered` is the default live/validation-safe mode. Each step uses only data visible at that step’s own as-of.
- `smoothed_research` is explicit retrospective mode. Each step uses a terminal retrospective as-of and requires `allow_research=True`.

## Engine contract

PIT transforms support two execution backends:

- `duckdb`: built-ins (`resample`, `aggregate`, `rolling`, `expanding`, `lag`, `diff`, `pct_change`) with supported parameters
- `python`: full operator coverage (including `ffill`, `coalesce`, `splice`, `path_apply`, and `binary`)

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

## Ref query contract

Typed ref-period PIT queries use:

- `RefSnapshotQuery`
- `RefRevisionQuery`

Public execution APIs:

- `PITAccessor.snapshot_ref(query)`
  - accepts `RefSnapshotQuery` or a mapping with equivalent fields
  - returns a `Series` indexed by typed `RefPeriod` values
  - requires explicit `freq` when both `start_ref` and `end_ref` are omitted
  - supports `obs_date_anchor="start" | "end"` for series whose stored
    observation dates are period-start or period-end keyed
- `PITAccessor.revisions_ref(query)`
  - accepts `RefRevisionQuery` or a mapping with equivalent fields
  - returns a revision timeline indexed by `asof_utc`
  - names the output with the canonical ref-entity id form

Compatibility wrappers:

- `get_snapshot_ref(...)`
- `get_revision_timeline_ref(...)`

remain supported during migration, but they are compatibility helpers rather
than the preferred public surface.

The nowcast-style contract coverage for this surface lives in
`tests/contracts/test_nowcast_pit_contract.py`.

Release stream helpers for reference periods:

- `list_release_stream(series_key, ref, asof=None, freq=None)`
  - returns one ref-period stream ordered by `asof_utc` with `release_rank`, `is_first`, and `is_latest`.
- `resolve_release(series_key, ref, policy=..., asof=None, freq=None)`
  - supports policies: `"first"`, `"latest"`, `{"mode":"rank","rank":n}`, `{"mode":"horizon","horizon":...}`.

## Series explainability contract

Persisted derived series can be inspected with:

- `get_series_lineage(series_key, start_obs=None, end_obs=None, start_asof=None, end_asof=None, limit=...)`
- `explain_series(series_key, start_obs=None, end_obs=None, start_asof=None, end_asof=None, limit=...)`

`get_series_lineage(...)` returns row-level provenance columns including:

- `lineage_kind`
- `transform_id`
- `graph_id`
- `node_name`
- `input_series_keys`
- `source_asof_utc`
- `selected_input_asof_utc`
- `source_asof_by_series_utc`
- `max_source_asof_utc`
- `causality_status`

`lineage_kind` values:

- `raw`
- `transform`
- `expression_graph`
- `derived` for other persisted lineage payloads

`causality_status` values:

- `raw`
- `ok`
- `unknown`
- `violation`
- `experimental`

`explain_series(...)` summarizes the row-level lineage into:

- unique input series keys
- transform ids
- expression graph ids
- row counts and derived-row counts
- aggregate causality status counts
- a boolean `causality_safe` summary flag

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
- `build_snapshot_panel_long(series_specs, asof, align=\"month_end|quarter_end\")`

Snapshot panel semantics:

- `get_snapshot_multi(...)` returns batch rows with:
  - `series_key`
  - `obs_date`
  - `source_asof_utc`
  - `value`
- `build_snapshot_panel_long(...)` returns aligned long rows with:
  - `series_key`
  - `series_alias`
  - `obs_date`
  - `source_obs_date`
  - `source_asof_utc`
  - `value`
- `build_snapshot_panel(...)` is the wide pivot over the aligned long form and
  preserves explicit `join` semantics (`inner | left | right | outer`)
- `SnapshotSeriesSpec` supports per-series `release_policy` plus optional
  `start_ref`, `end_ref`, `freq`, and `obs_date_anchor` for explicit ref-aware
  bounds
- panel alignment is deterministic and explicit; aligned panel dates do not
  discard the underlying source observation or source vintage metadata

## PIT contract versioning

Version API:

- `PIT_CONTRACT_VERSION`
- `get_pit_contract_version()`

Migration entries for contract/validation changes are recorded in `docs/guides/pit-migrations.md`.

## Data-source query contract

`PITDataSource` remains the legacy/raw-loader `DataSource` bridge into PIT
storage. Canonical PIT access lives on `PITAccessor`, but when a panel-style
integration still needs the `DataSource` contract, `PITDataSource` exposes
these table semantics:

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

`iter_walk_forward_folds` uses a type-narrowing assertion to guarantee `required_train` is a non-None `int` after the guard that requires at least one of `train_size` or `min_train_size` to be provided.
