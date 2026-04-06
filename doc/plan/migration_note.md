# Core Platform Migration Note

**Status:** Core platform epic complete; downstream migration active

## Purpose

This file is the running migration log for the core platform roadmap tracked
under `ALP-9`.

Use it to record downstream-impacting changes as implementation lands, with a
focus on:

- public API changes
- compatibility boundary changes
- downstream repo follow-ups
- required migration actions
- temporary shims and planned removals

This note complements the roadmap in
`doc/plan/core_platform_roadmap.md`. The roadmap explains what the program is
trying to build; this file records what changed and what downstream users need
to know.

Post-migration cleanup items that should only happen after downstream repos are
fully moved live in `doc/plan/post_migration_plan.md`.

## Downstream Repos To Watch

- `nowcast-data`
- `positioning`
- `steveya.github.io/posts/volatility-forecasts-*`

## Update Protocol

Add an entry whenever a ticket changes:

- the preferred public API
- the canonical loading path
- PIT query semantics
- dataset-spec or template behavior
- source health / archival semantics
- compatibility shims or deprecation plans

Each entry should include:

- date
- ticket
- summary of the change
- impacted public surface
- downstream repos affected
- migration action required, if any
- temporary compatibility path, if any
- follow-up notes

## Current Migration Fronts

These are the main areas where downstream code is expected to change as the
roadmap lands:

- ref-period and release-aware PIT semantics
- the canonical fetch path around `SourceAdapter`
- the reduced public role of `DataSource`
- lower-boilerplate loading and local context setup
- dataset-spec and template ergonomics
- release-aware health and archival helpers

## Current Canonical Paths

Downstream agents should treat these as the current preferred public surfaces.
The historical entries below explain how they got here; this section is the
fast path for current migration work.

### Temporal semantics

- import release rules and missingness from `alphaforge.time`
- use `RefPeriod`, `coerce_ref_period(...)`, `normalize_ref_freq(...)`, and
  `normalize_obs_date_anchor(...)` for explicit ref-period handling

### PIT queries and panels

- use `RefSnapshotQuery` and `RefRevisionQuery`
- use `PITAccessor.snapshot_ref(...)` and `PITAccessor.revisions_ref(...)`
- use `PITAccessor.build_snapshot_panel_long(...)` or
  `build_snapshot_panel(...)` for aligned PIT panels
- use `PITAccessor.get_series_lineage(...)` and
  `PITAccessor.explain_series(...)` for explainability

### Source access and loading

- use `SourceAdapter` as the canonical loading abstraction
- bootstrap with `DataContext.from_adapters(...)`
- load through `ctx.fetch(...)`, `ctx.fetch_many(...)`, `ctx.prefetch(...)`,
  and `ctx.load(...)`

### Dataset assembly and research UX

- use `DatasetSpec` with explicit `FeatureRequestGroup` composition where
  feature families belong together
- use built-in market templates from `alphaforge.features`, especially:
  - `LagReturnsTemplate`
  - `RollingVolatilityTemplate`

### Operations and source monitoring

- use `SourceHealthPolicy` with release-aware rules where needed
- use `build_health_report(...)` or `SourceHealthTracker.report(...)` for
  structured health output
- use `discover_archive_fetches(...)` and `iter_yearly_archive_fetches(...)`
  for deterministic archive planning

## Still-Temporary Compatibility Surfaces

These surfaces are still supported during downstream migration, but they should
not be treated as equal long-term public directions:

- `alphaforge.pit.release_rules`
- `alphaforge.pit.missingness`
- `PITAccessor.get_snapshot_ref(...)`
- `PITAccessor.get_revision_timeline_ref(...)`
- `alphaforge.pit.adapters.source_adapter_compat.SourceAdapterPITCompat`
- `DataContext.sources`
- `DataContext.fetch_panel(...)`
- boolean `strict=True/False` for `PITAccessor.upsert_pit_observations(...)`
- defaulting to `DataSource` as the primary public loading abstraction

Removal and cleanup of these bridges is tracked in
`doc/plan/post_migration_plan.md` under `ALP-23` through `ALP-27`.

## Repo-By-Repo Migration Checklist

Use these lists as the starting point for downstream migration work.

### `nowcast-data`

- move release-rule and missingness imports to `alphaforge.time`
- replace repo-local ref-period coercion with `RefPeriod` /
  `coerce_ref_period(...)`
- move ref-period PIT reads onto `RefSnapshotQuery` /
  `RefRevisionQuery` plus `snapshot_ref(...)` / `revisions_ref(...)`
- pass `obs_date_anchor` explicitly for period-start keyed series
- replace series-by-series PIT panel loops with shared panel builders
- prefer lineage APIs over direct `meta_json` parsing
- stop adding new dependencies on `get_snapshot_ref(...)` or
  `get_revision_timeline_ref(...)`

### `positioning`

- move new loading code onto `SourceAdapter` plus adapter-backed `DataContext`
- stop teaching `DataSource`, `ctx.sources[...]`, or `fetch_panel(...)` as the
  default public loading path
- use `ctx.fetch_many(...)` for batched loads instead of manual query loops
- adopt release-aware health reports via `build_health_report(...)` or
  `tracker.report(...)`
- adopt deterministic archive planning via `discover_archive_fetches(...)` or
  `iter_yearly_archive_fetches(...)`
- move temporal imports to `alphaforge.time` and prefer typed PIT helpers where
  PIT semantics are involved

### `steveya.github.io/posts/volatility-forecasts-*`

- move notebook recipes onto `DataContext.from_adapters(...)` and `ctx.load(...)`
- use `FeatureRequestGroup` for grouped feature families
- import `LagReturnsTemplate` and `RollingVolatilityTemplate` from
  `alphaforge.features`
- stop treating helper code under `examples/` as the reusable template API
- rely on `DatasetSpec` validation rather than permissive late failures for join
  and missingness policy strings

## Minimum Validation For Migration-Sensitive Changes

When migration work changes Alphaforge itself rather than only downstream call
sites:

- run `python -m pytest tests/contracts`
- rerun `python -m benchmarks.pit` when PIT retrieval performance might move
- update this file whenever the preferred public path, temporary bridge set, or
  downstream migration action changes

## Initial Setup Entry

### 2026-04-04

**Tickets:** `ALP-9` through `ALP-22`

**Summary:**

Started the durable execution scaffolding for the core platform roadmap:

- created the umbrella issue `ALP-9`
- created the full child ticket queue for roadmap implementation
- created this migration note
- updated the roadmap to treat migration guidance as a first-class workstream

**Impacted public surface:**

- None yet. This is planning and execution setup only.

**Downstream repos affected:**

- None yet. No implementation has landed.

**Migration action required:**

- None yet.

**Temporary compatibility path:**

- Existing downstream wrappers and compatibility layers remain unchanged.

**Follow-up notes:**

- The first migration-sensitive tickets are the temporal semantics, PIT API,
  data-access, and loading-ergonomics slices.
- As soon as one of those tickets lands, record the exact downstream behavior
  change here rather than only in PR or ticket commentary.

## Migration Entries

### 2026-04-04

**Ticket:** `ALP-10`

**Summary:**

Promoted release-rule and missingness semantics into the core
`alphaforge.time` package and made source health release-aware when a
`release_rule` is available.

**Impacted public surface:**

- canonical release-rule imports now live under `alphaforge.time`
- canonical missingness imports now live under `alphaforge.time`
- top-level `alphaforge` now re-exports the temporal semantic core
- `SourceHealthPolicy.release_rule` now changes health evaluation semantics
  from cadence-only aging to next-expected-release timing

**Downstream repos affected:**

- `nowcast-data`
- `positioning`

**Migration action required:**

- Prefer `from alphaforge.time import ReleaseRule, MissingnessReason,
  classify_missingness, ...` for new code.
- Existing health policies that set `release_rule` should expect status
  transitions to be measured against the next expected release window rather
  than only `latest_obs_date + expected_cadence`.

**Temporary compatibility path:**

- `alphaforge.pit.release_rules` and `alphaforge.pit.missingness` remain
  import-compatible shims.

**Follow-up notes:**

- `ALP-11` should build on this by tightening typed ref-period handling inside
  the same temporal-semantic layer.
- `ALP-20` can now rely on the shared release-aware health vocabulary instead
  of inventing separate operational timing rules.
- Shim removal after downstream migration is explicitly tracked in `ALP-23`
  and mirrored in `doc/plan/post_migration_plan.md`.

### 2026-04-04

**Ticket:** `ALP-11`

**Summary:**

Standardized ref-period normalization around the typed `alphaforge.time`
surface so PIT and target helpers share one explicit path for canonical ref
keys, pandas `Period` inputs, and explicit observation dates plus declared
frequency/anchor semantics.

**Impacted public surface:**

- `RefPeriod.parse(...)` now accepts typed normalization inputs beyond plain
  ref-key strings
- `alphaforge.time.ref_period` now exposes explicit normalization helpers such
  as `coerce_ref_period`, `normalize_ref_freq`, and `normalize_obs_date_anchor`
- PIT ref helpers now accept pandas `Period` inputs through the shared
  normalization path

**Downstream repos affected:**

- `nowcast-data`
- `positioning`

**Migration action required:**

- Prefer `RefPeriod` and `coerce_ref_period(...)` instead of repo-local quarter
  parsing or timestamp-to-ref conversions.
- When normalizing an observation date into a reference period, pass the
  intended frequency and anchor explicitly instead of relying on incidental date
  coercion.

**Temporary compatibility path:**

- Existing string ref keys such as `2024Q4` remain supported.
- No new legacy shim module was introduced in this ticket.

**Follow-up notes:**

- `ALP-12` can now build ref-period snapshot and revision APIs on top of the
  shared typed normalization surface instead of duplicating quarter/date logic.

### 2026-04-04

**Ticket:** `ALP-12`

**Summary:**

Added a first-class ref-period PIT query surface based on typed query objects
instead of asking downstream code to reach into legacy-style accessor helpers.

**Impacted public surface:**

- new canonical query objects:
  - `RefSnapshotQuery`
  - `RefRevisionQuery`
- new canonical execution methods:
  - `PITAccessor.snapshot_ref(...)`
  - `PITAccessor.revisions_ref(...)`
- `snapshot_ref(...)` now returns a `Series` indexed by typed `RefPeriod`
  values rather than raw observation timestamps
- explicit `obs_date_anchor` handling is now available on the canonical
  ref-query surface for period-start or period-end keyed series

**Downstream repos affected:**

- `nowcast-data`
- `positioning`

**Migration action required:**

- Prefer `snapshot_ref(...)` and `revisions_ref(...)` with typed query objects
  for new code.
- Prefer the RefPeriod-indexed snapshot output instead of repo-local wrappers
  that re-key observation timestamps back into reference periods.
- When a series stores ref observations at period start instead of period end,
  pass `obs_date_anchor="start"` explicitly in the query object.

**Temporary compatibility path:**

- `get_snapshot_ref(...)` and `get_revision_timeline_ref(...)` remain available
  as compatibility helpers during migration.

**Follow-up notes:**

- Post-migration cleanup for the legacy helper names is tracked in `ALP-27`
  and mirrored in `doc/plan/post_migration_plan.md`.
- `ALP-14` can build batch ref-period panel helpers on top of these typed query
  semantics instead of inventing another batch-only contract.

### 2026-04-04

**Ticket:** `ALP-14`

**Summary:**

Upgraded PIT batch retrieval and panel-building so common snapshot/panel
workloads can use a shared batch path with preserved source-vintage metadata
instead of series-by-series loops.

**Impacted public surface:**

- `get_snapshot_multi(...)` now returns `source_asof_utc`
- new canonical long helper:
  - `PITAccessor.build_snapshot_panel_long(...)`
- `SnapshotSeriesSpec` now accepts `freq` and `obs_date_anchor` for explicit
  ref-aware panel bounds
- `build_snapshot_panel(...)` now builds on the aligned long-form primitive

**Downstream repos affected:**

- `nowcast-data`
- `positioning`

**Migration action required:**

- Prefer `build_snapshot_panel(...)` or `build_snapshot_panel_long(...)` over
  repo-local loops that call `get_snapshot(...)` series-by-series.
- When downstream logic needs to retain the supplying vintage per panel row,
  consume `source_asof_utc` from `get_snapshot_multi(...)` or the long panel
  builder instead of reconstructing it manually.
- For period-start keyed ref series used inside panels, pass `freq` and
  `obs_date_anchor` explicitly in `SnapshotSeriesSpec`.

**Temporary compatibility path:**

- Existing single-series snapshot methods remain supported.
- No new compatibility-only shim module was introduced in this ticket.

**Follow-up notes:**

- `ALP-13` can build lineage and causal diagnostics on top of the preserved
  long-form source metadata that now comes out of the shared panel builder.

### 2026-04-04

**Ticket:** `ALP-13`

**Summary:**

Added public series-level lineage and causality inspection APIs so derived PIT
outputs can be explained directly from persisted storage instead of forcing
downstream code to parse `meta_json` by hand.

**Impacted public surface:**

- new lineage inspection API:
  - `PITAccessor.get_series_lineage(...)`
- new series summary API:
  - `PITAccessor.explain_series(...)`
- derived series now expose a normalized `causality_status` vocabulary in the
  public surface:
  - `raw`
  - `ok`
  - `unknown`
  - `violation`
  - `experimental`

**Downstream repos affected:**

- `nowcast-data`
- `positioning`

**Migration action required:**

- Prefer `get_series_lineage(...)` / `explain_series(...)` for persisted PIT
  provenance inspection instead of repo-local JSON parsing against `meta_json`.
- Prefer the public `causality_status` summary over ad hoc checks of
  `source_asof_utc` fields when inspecting whether a derived series is safe
  under the stored `asof_utc` semantics.

**Temporary compatibility path:**

- Existing lower-level access to raw `meta_json` remains available through the
  PIT table itself.
- No new compatibility shim or temporary bridge was introduced in this ticket.

**Follow-up notes:**

- `ALP-22` should carry these explainability APIs into the architecture and
  migration guides so downstream repos adopt the public inspection path rather
  than continuing to scrape raw lineage payloads.

### 2026-04-04

**Ticket:** `ALP-15`

**Summary:**

Canonicalized adapter-based source loading around `DataContext.fetch(...)`,
`fetch_many(...)`, and `prefetch(...)`, and tightened batch routing so
`fetch_many(...)` now delegates through the resolved adapter batch contract
instead of looping one query at a time.

**Impacted public surface:**

- canonical loading path is now explicitly:
  - `SourceAdapter`
  - `DataContext.fetch(...)`
  - `DataContext.fetch_many(...)`
  - `DataContext.prefetch(...)`
- `DataContext.fetch_many(...)` now batches by resolved adapter and preserves
  input order while forwarding `max_staleness`
- docs now treat `DataContext.sources` and `fetch_panel(...)` as
  backward-compatibility surfaces rather than co-equal public directions

**Downstream repos affected:**

- `positioning`
- `nowcast-data`
- `steveya.github.io/posts/volatility-forecasts-*`

**Migration action required:**

- Prefer adapter registration plus `ctx.fetch(...)` / `ctx.fetch_many(...)`
  for all new loading code.
- Stop teaching `ctx.sources[...]` or `ctx.fetch_panel(...)` as the default
  public loading path in downstream helper layers and notebooks.
- Where downstream code batches multiple queries manually, prefer
  `ctx.fetch_many(...)` so adapter-level cache and batch optimizations stay
  available.

**Temporary compatibility path:**

- `DataContext.sources` remains available for legacy `DataSource` users.
- `DataContext.fetch_panel(...)` remains available for legacy panel-oriented
  flows.

**Follow-up notes:**

- `ALP-16` should narrow the documented `DataSource` role even further now
  that the canonical fetch route is explicit and batch-capable.
- Final removal of the legacy context access path remains tracked in `ALP-25`
  and mirrored in `doc/plan/post_migration_plan.md`.

### 2026-04-04

**Ticket:** `ALP-16`

**Summary:**

Reduced `DataSource` to an explicitly documented compatibility/raw-loader role
instead of leaving it as an implied co-equal public loading contract alongside
`SourceAdapter`.

**Impacted public surface:**

- `alphaforge.data.source.DataSource` is now explicitly documented as a
  compatibility/raw-loader protocol
- `FREDDataSource` is now documented as a legacy loader, with
  `FREDSourceAdapter` as the preferred new-code path
- `PITDataSource` and the public-web loader docs now explicitly position
  `DataSource` usage as a bridge/raw-loader surface rather than the canonical
  general loading direction

**Downstream repos affected:**

- `positioning`
- `nowcast-data`
- `steveya.github.io/posts/volatility-forecasts-*`

**Migration action required:**

- Stop treating `DataSource` as the default public abstraction in downstream
  docs, helper layers, or new modules.
- Prefer `SourceAdapter` plus `DataContext.fetch(...)` for new loading code,
  even when older raw-loader integrations remain in the same repo.
- Keep direct `DataSource` usage only where a loader family is still
  intentionally raw-loader based or where an older panel/PIT integration has
  not migrated yet.

**Temporary compatibility path:**

- `DataSource` remains supported for public-web raw loaders, `PITDataSource`,
  and other legacy panel-oriented integrations.
- `DataContext.sources` and `fetch_panel(...)` remain available during the
  migration window.

**Follow-up notes:**

- `ALP-18` should build the short happy path on top of the now-clearly
  canonical adapter surface instead of papering over `DataSource` ambiguity.
- Eventual retirement of the remaining legacy context path is still tracked in
  `ALP-25`.

### 2026-04-04

**Ticket:** `ALP-17`

**Summary:**

Tightened the `DatasetSpec` contract around explicit feature-request
composition, early policy validation, and observable request metadata in the
feature catalog.

**Impacted public surface:**

- new typed composition surface:
  - `FeatureRequestGroup`
- `DatasetSpec.feature_requests()` now exposes the flattened request list used
  by the builder
- feature catalogs now include request/template metadata such as:
  - `request_key`
  - `template_name`
  - `template_version`
- `JoinPolicy` now validates `how` eagerly
- `MissingnessPolicy` now validates `final_row_policy` eagerly

**Downstream repos affected:**

- `steveya.github.io/posts/volatility-forecasts-*`
- `positioning`

**Migration action required:**

- Prefer `FeatureRequestGroup` when a notebook or research recipe has a family
  of related feature requests with shared tags, key prefixes, or slice
  overrides.
- Stop relying on invalid join or missingness strings making it deep into the
  builder; invalid values now fail at spec construction time.
- Where downstream code tracks feature families manually, prefer the stamped
  `request_key` and merged `tags_json` catalog fields.

**Temporary compatibility path:**

- Existing flat `features=[FeatureRequest(...), ...]` specs remain supported.
- No compatibility shim was introduced for this ticket.

**Follow-up notes:**

- `ALP-19` can now build notebook-ready recipes on top of `FeatureRequestGroup`
  instead of inventing another grouping abstraction.
- `ALP-22` should reflect the request-group composition model in the
  architecture and migration guides.

### 2026-04-04

**Ticket:** `ALP-18`

**Summary:**

Reduced happy-path loading boilerplate by adding adapter/bootstrap helpers for
common source-table and PIT read workflows.

**Impacted public surface:**

- new `DataContext.from_adapters(...)` classmethod for adapter-first bootstrap
- new `DataContext.load(...)` helper for common single-table loads without
  explicit `Query(...)` construction
- new `PITAccessor.open(path)` bootstrap for local DuckDB-backed PIT stores

**Downstream repos affected:**

- `positioning`
- `nowcast-data`
- `steveya.github.io/posts/volatility-forecasts-*`

**Migration action required:**

- Prefer `DataContext.from_adapters(...)` over manual `sources={}`,
  `adapters={...}`, and default-source boilerplate when wiring adapter-only
  contexts.
- Prefer `ctx.load(...)` for straightforward source-table reads where a custom
  `Query` object is not needed.
- Prefer `PITAccessor.open(path)` over manually instantiating
  `DuckDBParquetStore` and then passing `store.conn()` into `PITAccessor`.

**Temporary compatibility path:**

- Existing `ctx.fetch(Query(...))` calls remain supported.
- Existing `PITAccessor(DuckDBParquetStore(...).conn())` construction remains
  supported.

**Follow-up notes:**

- `ALP-21` should include contract tests around `from_adapters(...)`,
  `load(...)`, and `PITAccessor.open(...)` so future migrations do not regress
  the new short path.
- `ALP-22` should keep the happy-path examples synchronized with the ergonomic
  helper surface.

### 2026-04-05

**Ticket:** `ALP-19`

**Summary:**

Added canonical built-in market templates and recipe docs so volatility-style
research can use reusable lag-return and trailing-volatility feature families
instead of repo-local notebook helper cells.

**Impacted public surface:**

- new built-in market templates now live under `alphaforge.features`:
  - `LagReturnsTemplate`
  - `RollingVolatilityTemplate`
- the preferred market-research recipe path now uses:
  - `DataContext.from_adapters(...)`
  - `ctx.load(...)`
  - `FeatureRequestGroup`
  - built-in market templates rather than example-only helper modules
- quickstart and recipe docs now teach the adapter-backed volatility workflow
  directly

**Downstream repos affected:**

- `steveya.github.io/posts/volatility-forecasts-*`

**Migration action required:**

- Prefer `from alphaforge.features import LagReturnsTemplate,
  RollingVolatilityTemplate` for new research notebooks and template libraries.
- Prefer custom targets that build on `ctx.load(...)` rather than
  `ctx.fetch_panel(...)` when following the canonical adapter-backed dataset
  path.

**Temporary compatibility path:**

- Existing helper modules under `examples/` still exist for older demos, but
  they are no longer the preferred reusable import path for market features.

**Follow-up notes:**

- `ALP-21` should add contract coverage around the notebook-style volatility
  recipe so the new built-in template family remains stable during later
  compatibility cleanup.
- `ALP-22` should incorporate the new research recipe guidance into the
  architecture/migration closeout.

### 2026-04-05

**Ticket:** `ALP-20`

**Summary:**

Added a more operational release-aware health surface and upgraded archive
ingestion helpers from raw URL lists to deterministic fetch-plan entries.

**Impacted public surface:**

- `SourceHealthStatus` now carries structured release-aware diagnostics:
  - `overdue`
  - `overdue_days`
- new operational report helper:
  - `alphaforge.pipeline.health.build_health_report(...)`
- `SourceHealthTracker.report(...)` now returns the same dataframe-style health
  report and tracker persistence records `overdue_days`
- public-web archive helpers now expose deterministic fetch planning via:
  - `ArchiveFetchPlanEntry`
  - `discover_archive_fetches(...)`
  - `iter_yearly_archive_fetches(...)`
- archive-backed public-web loaders now use planned artifact names instead of
  open-coded URL-to-filename logic

**Downstream repos affected:**

- `positioning`

**Migration action required:**

- Prefer structured `overdue_days` and `build_health_report(...)` /
  `tracker.report(...)` over parsing release-aware delay information out of
  free-form status messages.
- Prefer `discover_archive_fetches(...)` or `iter_yearly_archive_fetches(...)`
  when building recurring archive ingestion flows so artifact names and year
  metadata stay deterministic.

**Temporary compatibility path:**

- Existing low-level URL helpers (`discover_archive_links(...)`,
  `filter_urls_for_years(...)`, `iter_yearly_archive_urls(...)`) remain
  available for callers that intentionally want raw URL primitives.

**Follow-up notes:**

- `ALP-21` should lock down the release-aware report shape and archive fetch
  planning semantics as part of the downstream-inspired contract suite.

### 2026-04-05

**Ticket:** `ALP-21`

**Summary:**

Added an explicit downstream contract suite plus a PIT benchmark harness so
future public-surface changes can be validated against the workflows that
already depend on Alphaforge.

**Impacted public surface:**

- new contract-test slice:
  - `tests/contracts/test_nowcast_pit_contract.py`
  - `tests/contracts/test_volatility_dataset_contract.py`
  - `tests/contracts/test_operations_contract.py`
  - `tests/contracts/test_pit_benchmarks.py`
- new benchmark harness:
  - `python -m benchmarks.pit`
  - `benchmarks.run_pit_contract_benchmarks(...)`
- the PIT contract docs now treat the downstream contract slice and benchmark
  harness as migration gates for compatibility-sensitive work

**Downstream repos affected:**

- `nowcast-data`
- `positioning`
- `steveya.github.io/posts/volatility-forecasts-*`

**Migration action required:**

- Run `python -m pytest tests/contracts` when changing the canonical PIT,
  adapter/data-context, dataset-spec/template, or operational source surfaces.
- Rerun `python -m benchmarks.pit` when changing PIT retrieval paths so
  benchmark snapshots stay current in the migration discussion.
- Use the contract files as explicit evidence when removing or narrowing a
  compatibility surface.

**Temporary compatibility path:**

- No new compatibility shim was introduced in this ticket.
- The contract suite exists to police previously introduced compatibility
  bridges rather than to add another one.

**Follow-up notes:**

- `ALP-22` should fold these regression gates into the architecture and
  migration guides so downstream maintainers can see one coherent story.
- Current PIT benchmark baseline captured at
  `2026-04-05T03:35:01.847144+00:00` with
  `python -m benchmarks.pit --iterations 5 --periods 40 --series-count 3 --revisions-per-period 2`:
  `snapshot_ref(...)` median 5.724 ms,
  `build_snapshot_panel_long(...)` median 11.518 ms.

### 2026-04-05

**Ticket:** `ALP-22`

**Summary:**

Published explicit core-platform architecture and migration guides and kept the
repo-local roadmap and migration logs synchronized through the end of the epic.

**Impacted public surface:**

- new architecture guide for canonical layer boundaries
- new migration guide for downstream moves onto canonical public paths
- new contracts-and-benchmarks guide for the stability discipline behind
  migration and deprecation work
- development docs now call out the roadmap-specific regression gates

**Downstream repos affected:**

- `nowcast-data`
- `positioning`
- `steveya.github.io/posts/volatility-forecasts-*`

**Migration action required:**

- Use the architecture guide as the canonical map of layer ownership and public
  direction.
- Use the migration guide when moving code off legacy PIT helpers, legacy
  `DataContext` access, and temporary compatibility imports.
- Keep future downstream-impacting tickets updating this note rather than
  relying on scattered PR context.

**Temporary compatibility path:**

- No new compatibility shim or temporary bridge was introduced in this ticket.
- Existing migration bridges remain tracked in
  `doc/plan/post_migration_plan.md`.

**Follow-up notes:**

- With the core platform epic landed, future work should treat these guides as
  the canonical documentation entry points for architecture and migration.

## Maintenance Queue

### 2026-04-05

| Ticket | Title | Status |
| --- | --- | --- |
| `ALP-28` | `Data access: fail on ambiguous adapter routing without a default source` | Done |
| `ALP-29` | `Temporal semantics: honor WeeklyRelease weekday configuration` | Done |
| `ALP-30` | `CFTC adapter: preserve disaggregated PIT source lineage` | Done |
| `ALP-31` | `Public web: surface CFTC archive fetch failures instead of silently dropping years` | Done |
| `ALP-32` | `Canonical loading: migrate public-web outliers onto adapter-backed access` | Open |
| `ALP-33` | `DTCC adapters: add shared product-family adapter base over DTCC PPD raw loader` | Done |
| `ALP-34` | `DTCC adapters: add first product-family adapters and dataset contracts` | Done |
| `ALP-35` | `MOF JGB: add adapter-backed canonical load path for constant-maturity yields` | Open |
| `ALP-36` | `Philadelphia SPF: add adapter-backed canonical load path for mean-level surveys` | Open |
| `ALP-37` | `Research UX: add canonical wide curve and maturity-order helpers for adapter-backed loads` | Open |
| `ALP-38` | `Migration: move short_rates and notebook examples off raw-loader outlier APIs` | Blocked by `ALP-35`, `ALP-36`, `ALP-37` |

### 2026-04-05

**Tickets:** `ALP-28`, `ALP-29`, `ALP-30`, `ALP-31`

**Summary:**

Hardened four migration-sensitive public surfaces:

- canonical adapter routing now fails on ambiguous shared datasets unless a
  default or explicit source is provided
- `WeeklyRelease` now honors its configured weekday instead of behaving like a
  raw lag-only offset
- the multi-dataset CFTC adapter now preserves distinct PIT lineage for
  disaggregated CoT rows
- CFTC archive-backed public-web loads now fail fast on broken requested
  archives instead of silently returning partial history

**Impacted public surface:**

- `DataContext.from_adapters(...)`, `ctx.fetch(...)`, and `ctx.load(...)` for
  datasets served by more than one adapter
- `WeeklyRelease.expected_release_date(...)` and release-aware health
  semantics that consume weekly rules
- CFTC PIT row provenance for `cot.disagg`
- CFTC public-web archive fetch behavior and operational error visibility

**Downstream repos affected:**

- `positioning`
- `nowcast-data`

**Migration action required:**

- downstream callers that relied on adapter registration order for shared
  datasets must now configure `default_sources` or pass `source=`
- weekly release schedules should expect the configured weekday to matter after
  the lag anchor
- workflows that previously tolerated silently partial CFTC archive history
  should now expect an explicit failure and handle or fix the broken archive
  instead

**Temporary compatibility path:**

- No new compatibility shim was introduced.
- These tickets narrow ambiguous behavior on canonical paths rather than
  creating another bridge.

**Follow-up notes:**

- Existing cached `cot.disagg` PIT rows written before `ALP-30` keep their old
  lineage until repopulated.
- The next public-web maintenance pass should decide whether other
  archive-backed loaders should adopt the same fail-fast behavior as `ALP-31`.

### 2026-04-05

**Ticket:** `ALP-33`

**Summary:**

Introduced a shared DTCC adapter base above `DTCCPPDSource` so canonical DTCC
adapters can own raw-loader wiring internally instead of forcing callers to
inject ad hoc raw-fetch closures.

**Impacted public surface:**

- new shared DTCC adapter base:
  - `DTCCPPDAdapterBase`
- `DTCCAdapter` now constructs and owns `DTCCPPDSource` internally by default
- future DTCC product-family adapters can vary dataset contracts and PIT
  transforms without duplicating cache, prefetch, and raw-query plumbing

**Downstream repos affected:**

- `positioning`
- `steveya.github.io`

**Migration action required:**

- Prefer `DTCCAdapter(...)` with raw-source keyword arguments instead of
  building raw-fetcher closures around `DTCCPPDSource`.
- When adding DTCC canonical datasets such as FX options or IRS, subclass
  `DTCCPPDAdapterBase` rather than copying `DTCCAdapter` cache/fetch logic.
- Keep direct `DTCCPPDSource` usage only on the raw-loader compatibility side
  of the API boundary.

**Temporary compatibility path:**

- Tests and advanced callers may still inject a preconfigured `source=`
  object into `DTCCAdapter`.
- Direct `DTCCPPDSource` usage remains available through the public-web
  raw-loader surface.

**Follow-up notes:**

- `ALP-34` should define the first concrete DTCC product-family adapters and
  their dataset contracts on top of `DTCCPPDAdapterBase`.
- The remaining outlier canonicalization work for MOF, SPF, and notebook-style
  wide helpers remains tracked in `ALP-35` through `ALP-38`.

### 2026-04-05

**Ticket:** `ALP-34`

**Summary:**

Added the first concrete DTCC product-family adapters so the preferred DTCC
adapter surface is no longer one generic `dtcc.ppd` dataset.

**Impacted public surface:**

- new canonical DTCC family adapters:
  - `DTCCFXAdapter` with dataset `dtcc.fx`
  - `DTCCIRSAdapter` with dataset `dtcc.irs`
- `dtcc_daily_to_pit_observations(...)` now accepts custom `key_prefix` and
  `source_name` so family adapters can stamp distinct series keys and PIT
  lineage without copying the transform
- `DTCCAdapter` remains available as the broader generic wrapper, but canonical
  docs now teach `dtcc.fx` and `dtcc.irs` first

**Downstream repos affected:**

- `positioning`
- `steveya.github.io`

**Migration action required:**

- Prefer `DataContext.from_adapters(DTCCFXAdapter(...), DTCCIRSAdapter(...))`
  plus `ctx.load("dtcc.fx", ...)` / `ctx.load("dtcc.irs", ...)` for new DTCC
  research or ingestion code.
- Stop introducing new downstream dependencies on the generic `dtcc.ppd`
  dataset when the workflow is specifically FX or IRS.
- Keep direct `DTCCPPDSource` usage only where the raw-loader compatibility
  surface is explicitly intended.

**Temporary compatibility path:**

- `DTCCAdapter` still serves the broader `dtcc.ppd` dataset.
- `DTCCPPDSource` remains available as the low-level raw-loader surface.

**Follow-up notes:**

- `DTCCFXAdapter` currently covers FX forwards and swaps; it is not yet an
  FX-options-specific contract because current provider fixtures do not expose
  that family cleanly.
- `DTCCIRSAdapter` intentionally excludes OIS and cross-currency swaps even
  though they share the same low-level IR artifact family.
- Future DTCC slices can add more family adapters on top of the filtered-family
  pattern without reopening the raw-loader ownership work from `ALP-33`.
