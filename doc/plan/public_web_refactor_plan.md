# `alphaforge.data.public_web` Refactor Plan

## Goal

Refactor `alphaforge.data.public_web` to reduce repeated source boilerplate, make new loaders cheaper to add, and centralize the patterns that are already shared in practice without forcing unrelated loaders into an artificial hierarchy.

This plan is intentionally incremental. The module contains several loader families, but it also contains true outliers. The refactor should extract stable abstractions around the shared fetch pipeline, not attempt to make every source look identical.

## Current State

Across the module, most loaders repeat the same high-level flow:

1. Build or accept a `CachedHttpClient`.
2. Validate `q.table`.
3. Download one or more artifacts.
4. Parse bytes into `DataFrame` objects.
5. Normalize dates, entity ids, and values.
6. Build an output frame with `date` / `entity_id` / `asof_utc`.
7. Apply `apply_query_filters(...)`.
8. Apply `project_columns(...)`.
9. Return either an empty schema-compatible frame or a sorted normalized frame.

That pattern is visible in many current loaders, including:

- `bea.py`
- `eia.py`
- `bcb_sgs.py`
- `eurostat.py`
- `destatis_genesis.py`
- `ibge_sidra.py`
- `eurex_stats_daily.py`
- `lch_cdsclear_daily.py`
- `ec_weekly_oil_bulletin.py`
- `cftc_swaps_weekly.py`
- `cftc_cot.py`
- `b3_historical_quotes.py`

There are also clear source families:

### 1. Registry-driven HTTP API sources

These sources load entity metadata from YAML registries, iterate requested entities, call an API per entity/config, then map provider-specific payloads into the standard long frame.

Candidates:

- `bea.py`
- `eia.py`
- `ecb_sdmx.py`
- `eurostat.py`
- `ibge_sidra.py`
- `destatis_genesis.py`

### 2. Simple single-endpoint API sources

These do not use registries but still follow the same "call endpoint -> normalize frame -> finalize" model.

Candidates:

- `bcb_sgs.py`
- `bls.py`

### 3. Tabular document sources

These fetch HTML / CSV / XLSX / ZIP artifacts, parse tables, identify relevant columns, then normalize them.

Candidates:

- `eurex_stats_daily.py`
- `lch_cdsclear_daily.py`
- `ec_weekly_oil_bulletin.py`
- `ezoic_adrevenue_daily.py`
- `cme_productslate_reference.py`
- `eurex_refdata_contracts.py`
- `frb_term_structure.py`

### 4. Archive / bulk-file sources

These discover or generate one or more artifact URLs, download archives, parse them, and normalize the combined result.

Candidates:

- `cftc_cot.py`
- `cftc_swaps_weekly.py`
- `b3_historical_quotes.py`

### 5. Complex outliers

These should likely adopt only the shallow shared helpers, not a deep family base class.

Candidates:

- `dtcc_ppd.py`
- `mof_jgb.py`
- `philadelphia_spf.py`

## Design Principles

### Prefer composable helpers over deep inheritance

The current module is diverse enough that a large abstract base class would become brittle. The better target is:

- small shared helper functions
- narrow mixins / base classes for clear source families
- explicit per-source normalization logic

### Keep one source per file

The refactor should not collapse unrelated loaders into generic frameworks. The implementation logic for each provider should remain discoverable in its own file.

### Standardize the outer fetch pipeline

The biggest immediate win is not parsing logic reuse. It is making the outer source lifecycle consistent:

- table validation
- empty-frame construction
- `asof_utc` defaults
- common filter/project/finalize behavior
- common HTTP wiring

### Avoid a mandatory "generic DSL"

Do not introduce a declarative framework that every source must conform to. A small helper library is lower-risk and more maintainable than a large meta-source layer.

## Proposed Target Structure

Introduce a small internal foundation layer under `alphaforge/data/public_web/`:

- `base.py`
- `schema_helpers.py`
- `finalize.py`
- `tabular.py`
- `registry_api.py`
- `archive.py`

These names are suggestions, not requirements. The key point is to separate:

- fetch lifecycle helpers
- schema/empty-frame helpers
- parsing/document helpers
- registry-backed API iteration helpers
- archive/batch helpers

## Proposed Abstractions

## A. `PublicWebSourceBase`

Add a light shared base for HTTP-backed sources. This should solve only the repeated shell around fetching.

Responsibilities:

- initialize / inject `CachedHttpClient`
- expose `self._now_utc()`
- validate `q.table`
- create empty frames from schema metadata
- centralize the final `apply_query_filters` + `project_columns` step

Suggested interface:

```python
class PublicWebSourceBase(DataSource):
    name: str

    def _require_table(self, q: Query, expected: str) -> None: ...
    def _empty_frame(self, schema: TableSchema) -> pd.DataFrame: ...
    def _finalize(
        self,
        df: pd.DataFrame,
        *,
        q: Query,
        schema: TableSchema,
        time_col: str | None = None,
        entity_col: str | None = None,
        sort_by: list[str] | None = None,
    ) -> pd.DataFrame: ...
    def _now_utc(self) -> pd.Timestamp: ...
```

This should replace repeated source-level code, not provider-specific parsing code.

## B. Schema helpers

Common `TableSchema` construction is repeated across the module, especially for:

- single-value macro series
- daily market tables
- event tables
- weekly interval-end tables

Add small helper constructors such as:

- `single_value_schema(...)`
- `daily_panel_schema(...)`
- `event_table_schema(...)`

This is mostly a readability improvement and should be kept shallow.

## C. Finalization helpers

Create a common helper for the dominant pattern:

```python
out = apply_query_filters(...)
out = project_columns(...)
return out.sort_values(...).reset_index(drop=True)
```

This should live in one place rather than being repeated in nearly every source.

It should also standardize:

- empty result columns
- `asof_utc` presence
- default sorting

## D. Registry-backed API base

Several sources share a very similar shape:

- load YAML registry
- require `q.entities`
- look up entity config
- call remote API per entity
- parse provider payload per entity
- assemble standardized rows

Add a narrow base or helper like:

```python
class RegistryApiSourceBase(PublicWebSourceBase):
    def _load_registry(...)
    def _iter_entity_configs(...)
```

Likely adopters:

- `bea.py`
- `eia.py`
- `ecb_sdmx.py`
- `eurostat.py`
- `ibge_sidra.py`
- `destatis_genesis.py`

Important constraint:

This base should not try to standardize provider payload shapes. It should only standardize registry loading, entity iteration, and row accumulation.

## E. Tabular document helpers

Multiple sources do:

- fetch bytes
- parse HTML tables or ZIP/CSV/XLSX sheets
- detect date/entity/metric columns using `first_existing(...)`
- normalize into standard columns

Add shared utilities for:

- selecting candidate tables by required columns
- resolving date columns from a list of aliases
- resolving metric columns from alias sets
- turning a normalized intermediate frame into a finalized output frame

This is especially relevant for:

- `eurex_stats_daily.py`
- `lch_cdsclear_daily.py`
- `ec_weekly_oil_bulletin.py`
- `ezoic_adrevenue_daily.py`

## F. Archive / batch helpers

Add helpers for sources that work from year-partitioned or discovered artifact lists:

- static yearly URL generation
- optional historical-batch URL inclusion
- ZIP member selection
- concatenation and de-duplication

This would directly benefit:

- `cftc_cot.py`
- `cftc_swaps_weekly.py`
- `b3_historical_quotes.py`

The recent CoT refactor already demonstrates the value of this direction.

## G. Naming and contract cleanup

Standardize the internal conventions used by all sources:

- `TABLE` for single-table sources
- `*_TABLE` for multi-table sources
- `name` always equal to registry/source key
- `date` for interval-end / observation date
- `ts_utc` for event timestamps only
- `entity_id` unless the schema intentionally exposes a different entity column

Also standardize the internal helper naming:

- `_call(...)` for remote API calls
- `_read_*` for artifact parsing
- `_normalize_*` for provider-to-canonical transformation
- `_discover_*` or `_list_*` for remote artifact enumeration

## Recommended Phases

## Phase 1: Extract the safe shallow helpers

Scope:

- add `PublicWebSourceBase`
- add finalization helpers
- add schema helpers

Do not change provider logic yet.

Candidate first adopters:

- `bcb_sgs.py`
- `bea.py`
- `eia.py`

Success condition:

- these sources get materially smaller without semantic changes
- tests remain unchanged except for helper-specific coverage

## Phase 2: Registry-backed API family

Scope:

- add `RegistryApiSourceBase`
- migrate registry-driven loaders to the new base

Candidate order:

1. `eia.py`
2. `bea.py`
3. `eurostat.py`
4. `ecb_sdmx.py`
5. `ibge_sidra.py`
6. `destatis_genesis.py`

Success condition:

- registry loading / entity iteration duplication is eliminated
- per-provider parsing remains local to each file

## Phase 3: Tabular-document family

Scope:

- add table-selection and alias-resolution helpers
- migrate HTML/table-driven sources

Candidate order:

1. `eurex_stats_daily.py`
2. `lch_cdsclear_daily.py`
3. `ec_weekly_oil_bulletin.py`
4. `ezoic_adrevenue_daily.py`

Success condition:

- repeated `first_existing(...)` / `ensure_date_utc(...)` / `entity_id` construction logic is reduced
- source-specific column semantics still remain explicit

## Phase 4: Archive / batch family

Scope:

- extract shared archive helpers
- migrate `cftc_swaps_weekly.py` and `b3_historical_quotes.py`
- keep `cftc_cot.py` as the reference implementation for the family

Success condition:

- URL generation, historical-batch support, ZIP reading, and empty-frame behavior are standardized

## Phase 5: Outlier integration

Scope:

- apply only shallow helper adoption to `dtcc_ppd.py`, `mof_jgb.py`, and `philadelphia_spf.py`

These should likely use:

- common HTTP setup
- common finalization
- common empty-frame helpers

They should probably not be forced into the family bases from phases 2-4.

## Phase 6: Public module cleanup

Scope:

- clean import/export organization in `__init__.py`
- keep registry construction simple in `registry.py`
- update developer docs for how to add a new source

Deliverables:

- a short "how to add a public_web source" document
- explicit source family guidance

## Risks

### 1. Over-abstraction

Risk:

Trying to unify all sources under one base class will make the abstraction worse than the current repetition.

Mitigation:

- keep bases shallow
- use family-level helpers only where there are at least 3 strong adopters
- treat `dtcc_ppd.py`, `mof_jgb.py`, and `philadelphia_spf.py` as exceptions by default

### 2. Silent output drift

Risk:

Refactoring the outer fetch flow can change empty-frame columns, sorting, time normalization, or `asof_utc` behavior.

Mitigation:

- add focused tests around output columns and sort order before migration
- migrate a few sources at a time
- preserve existing schema contracts exactly

### 3. Test churn without value

Risk:

Changing too many files at once can produce large low-signal diffs.

Mitigation:

- phase the work
- land family-level helpers first
- migrate only a small source set per PR

### 4. Forcing registry-backed assumptions into non-registry sources

Risk:

Registry-backed sources and bulk archive sources have very different query shapes.

Mitigation:

- separate the bases
- do not let `RegistryApiSourceBase` leak into archive or document loaders

## Suggested Initial PR Breakdown

### PR 1

- add `PublicWebSourceBase`
- add finalization helpers
- migrate `bcb_sgs.py`, `bea.py`, `eia.py`

### PR 2

- add `RegistryApiSourceBase`
- migrate remaining registry-driven sources

### PR 3

- add tabular helpers
- migrate `eurex_stats_daily.py`, `lch_cdsclear_daily.py`, `ec_weekly_oil_bulletin.py`

### PR 4

- add archive helpers
- migrate `cftc_swaps_weekly.py`, `b3_historical_quotes.py`

### PR 5

- shallow helper adoption for `dtcc_ppd.py`, `mof_jgb.py`, `philadelphia_spf.py`
- docs update

## Linear Ticket Mirror

These tables mirror the current Linear public-web refactor tickets and define
the intended implementation order for subsequent agents.

Linear routing for this plan:

- Team: `ALP` (`alphaforge`)
- Project: `alphaforge`
- Umbrella issue: `ALP-1`

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This plan file is the repo-local execution mirror for subsequent agents.
- Implement the earliest ticket in the ordered queue below whose status is not
  `Done` and whose upstream prerequisites are already satisfied.
- Skip tickets whose table row is `Done`.
- Epic rows are tracking rows. Do not pick them up before their earlier child
  rows unless the epic has no remaining open child slices.
- Before coding, print the current ticket number and its plain-English goal on
  screen.
- Follow the implementation workflow in `AGENTS.md`:
  - review upstream tickets and notes first
  - implement with TDD
  - update docs after tests pass
  - leave the structured closeout note in Linear
  - mark the Linear ticket `Done`
  - only then update the corresponding row in this file
- Update this table only after the ticket is closed in Linear.

Status mirror last synced: `2026-04-04`

### Ordered Epic Queue

| Ticket | Status |
| --- | --- |
| `ALP-1` Public web: refactor shared abstractions and source-family cleanup | Done |

### Ordered Public Web Refactor Queue

| Ticket | Status |
| --- | --- |
| `ALP-2` Public web foundation: add shared source base, schema helpers, and finalization helpers | Done |
| `ALP-3` Registry APIs: add a shared base and migrate registry-backed public sources | Done |
| `ALP-4` Simple APIs: migrate remaining single-endpoint public sources onto shared helpers | Done |
| `ALP-5` Tabular loaders: extract document helpers and migrate table-driven public sources | Done |
| `ALP-6` Archive loaders: extract batch helpers and migrate bulk-file public sources | Done |
| `ALP-7` Public web outliers: adopt shallow shared helpers in complex sources | Done |
| `ALP-8` Public web docs: add source-authoring guidance and complete module cleanup | Done |

### Cross-Ticket Sequencing Constraints

Use the queue order above, but also respect these concrete handoff rules:

- Treat `ALP-1` as a tracking epic. Start implementation with `ALP-2`.
- Do not start `ALP-3`, `ALP-4`, `ALP-5`, `ALP-6`, or `ALP-7` until `ALP-2` is
  landed.
- Treat `ALP-3`, `ALP-4`, `ALP-5`, `ALP-6`, and `ALP-7` as an ordered
  implementation queue even though the Linear dependency graph only records the
  shared blocker on `ALP-2`.
- Use `ALP-3` before `ALP-4` so the registry-backed family is migrated before
  the remaining simple API cleanup slice.
- Use `ALP-5` before `ALP-6` so the table/document helper layer lands before
  the archive/batch helper layer.
- Use `ALP-7` only after the family migrations are done; it is a shallow
  cleanup pass for true outliers, not a substitute for the family abstractions.
- Do not start `ALP-8` until `ALP-2`, `ALP-3`, `ALP-4`, `ALP-5`, `ALP-6`, and
  `ALP-7` are landed.

### Agent Pickup Directive

If you hand this file to a coding agent for end-to-end implementation, the
agent should:

1. start at the top of the epic queue
2. within the active queue, pick the first non-`Done` ticket whose
   prerequisites are satisfied
3. implement only that ticket's scoped outcome
4. close the ticket in Linear first
5. then update the mirrored status row in this plan before moving on

Do not skip ahead to later slices because a later ticket looks smaller. This
plan is intentionally ordered to land the shared foundation first, then the
clear source families, then the outlier/doc cleanup work.

## Epic Close-Out

The epic is complete.

Delivered outcomes:

- shared foundation helpers in `base.py`, `finalize.py`, and `schema_helpers.py`
- narrow family helpers for registry-backed APIs, tabular/document loaders, and
  archive/batch loaders
- shallow helper adoption in the outlier sources without forcing deeper
  inheritance
- public-web authoring guidance, API docs, quickstart updates, and package
  export cleanup

Final validation:

- `/Users/steveyang/miniforge3/bin/python -m pytest tests/test_public_web_registry_exports.py tests/public_web -k 'not live_sources' tests/test_cftc_dtcc_adapter.py -q`
- `/Users/steveyang/miniforge3/bin/python -m ruff check .`
- `/Users/steveyang/miniforge3/bin/python -m mkdocs build --strict`

## Success Criteria

The refactor is successful if:

- new public-web loaders can be added with materially less boilerplate
- common fetch/finalize behavior is centralized
- registry-backed and archive-backed families have explicit support
- complex outliers are not damaged by the abstraction effort
- no public schemas or table names change
- source tests remain one-to-one with modules and retain current behavior

## Non-Goals

- rewriting all loaders into a declarative framework
- changing public table names or entity-id contracts
- merging unrelated source files
- replacing pandas-based parsing with a different execution model
- changing `CachedHttpClient` transport semantics in the same refactor unless required by a separate bug fix

## Immediate Recommendation

Start with Phase 1 only. The module already has enough evidence that shallow fetch/finalize abstraction is worth it. That first slice should produce a smaller diff, validate the direction, and make the later family-specific abstractions easier to judge with real code instead of speculation.
