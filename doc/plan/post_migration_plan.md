# Post-Migration Plan

**Status:** Active backlog

## Purpose

Track cleanup work that should happen only after downstream migrations are
complete, so temporary compatibility paths do not become permanent by default.

This file is intentionally small and explicit. Add entries here when a roadmap
slice introduces a temporary shim, deprecation bridge, or compatibility-only
surface that should be removed once migration is done.

## Update Notes

### 2026-04-04

- `ALP-11` added no new post-migration cleanup artifact. The ref-period work
  extended the canonical `alphaforge.time` surface and reused existing string
  compatibility without introducing a new shim or temporary bridge.
- `ALP-12` introduced canonical typed ref-query APIs and kept
  `get_snapshot_ref(...)` / `get_revision_timeline_ref(...)` as temporary
  compatibility helpers. Their cleanup is tracked in `ALP-27`.
- `ALP-14` added no new post-migration cleanup artifact. It widened the
  canonical PIT batch/panel surface and upgraded existing snapshot metadata,
  but it did not introduce a new temporary bridge or shim.
- `ALP-13` added no new post-migration cleanup artifact. It exposed public
  lineage and causality inspection APIs over existing persisted metadata
  without adding a temporary compatibility layer.
- `ALP-15` added no new post-migration cleanup artifact. It made the
  `SourceAdapter` plus `DataContext.fetch(...)` / `fetch_many(...)` route the
  canonical public path and strengthened the migration rationale for `ALP-25`,
  but it did not introduce an additional shim or bridge.
- `ALP-16` added no new post-migration cleanup artifact. It narrowed
  `DataSource` to an explicitly documented compatibility/raw-loader role and
  further clarified why the remaining legacy context surface should eventually
  be retired under `ALP-25`, but it did not introduce another temporary shim.
- `ALP-17` added no new post-migration cleanup artifact. It tightened the
  canonical `DatasetSpec` surface and introduced `FeatureRequestGroup` as part
  of that public contract, not as a temporary compatibility bridge.
- `ALP-18` added no new post-migration cleanup artifact. It introduced
  ergonomic bootstrap helpers (`DataContext.from_adapters(...)`,
  `DataContext.load(...)`, `PITAccessor.open(...)`) as part of the intended
  long-term public surface rather than as temporary shims.
- `ALP-19` added no new post-migration cleanup artifact. It promoted
  notebook-ready market templates into the canonical `alphaforge.features`
  surface and updated docs/examples toward that path, but it did not introduce
  a temporary bridge or shim that should later be removed.
- `ALP-20` added no new post-migration cleanup artifact. The new health-report
  helpers and archive fetch-plan objects are intended long-term operational
  surfaces, while the older raw URL helpers remain useful lower-level
  primitives rather than temporary migration shims.
- `ALP-21` added no new post-migration cleanup artifact. It introduced
  contract tests and a PIT benchmark harness to police existing compatibility
  bridges, but it did not create a new temporary surface that should later be
  removed.
- `ALP-22` added no new post-migration cleanup artifact. It published the
  architecture and migration guides that explain the canonical versus legacy
  split, but it did not introduce another compatibility bridge.

## Active Post-Migration Queue

### 2026-04-04

**Ticket:** `ALP-23`

**Title:**

`Platform cleanup: remove temporal-semantics compatibility shims after migration`

**Trigger:**

- `nowcast-data`
- `positioning`
- other supported downstream consumers

must no longer import:

- `alphaforge.pit.release_rules`
- `alphaforge.pit.missingness`

**Cleanup to perform:**

- remove the compatibility shim modules at:
  - `alphaforge/pit/release_rules.py`
  - `alphaforge/pit/missingness.py`
- remove any remaining internal imports that still rely on those legacy PIT
  paths
- update migration docs and release notes to reflect shim removal
- keep `alphaforge.time` as the only supported public path for release rules
  and missingness

**Why it exists:**

`ALP-10` deliberately kept these shims to avoid breaking downstream consumers
during migration. They add short-term migration value, but they also preserve
API ambiguity. They should be removed once the downstream migration window is
closed.

**Validation at cleanup time:**

- targeted import-path regression tests
- relevant downstream compatibility checks
- `ruff check .`
- relevant `pytest` slices

### 2026-04-04

**Ticket:** `ALP-24`

**Title:**

`Platform cleanup: remove SourceAdapterPITCompat bridge after PIT adapter migration`

**Trigger:**

- `nowcast-data`
- any other supported PIT consumer

must no longer require the legacy `PITAdapter` interface or the bridge at:

- `alphaforge.pit.adapters.source_adapter_compat.SourceAdapterPITCompat`

**Cleanup to perform:**

- remove `alphaforge/pit/adapters/source_adapter_compat.py`
- remove API docs that present the bridge as a supported public path
- migrate any remaining internal callers to the canonical adapter-based path
- remove bridge-specific compatibility tests and fixtures

**Why it exists:**

The bridge is an intentional migration aid from legacy `PITAdapter` consumers
to the unified `SourceAdapter` layer. It is useful during migration, but it
keeps a parallel adapter model alive and should not remain indefinitely.

**Validation at cleanup time:**

- targeted PIT adapter migration checks
- relevant downstream compatibility checks
- `ruff check .`
- relevant `pytest` slices

### 2026-04-04

**Ticket:** `ALP-25`

**Title:**

`Platform cleanup: remove legacy DataContext source access after adapter migration`

**Trigger:**

- public docs and supported downstream consumers

must no longer rely on:

- `DataContext.sources`
- `DataContext.fetch_panel(...)`
- `ctx.sources[...]` as a user-facing loading pattern

**Cleanup to perform:**

- remove or retire the legacy `sources` mapping from the supported public path
- remove or retire `fetch_panel(...)` as a supported legacy access path
- update docs, examples, and tests that still teach or rely on the legacy path
- keep any remaining raw-loader internals only if they are explicitly treated
  as non-public

**Why it exists:**

The canonical data-access direction is moving toward `SourceAdapter` plus
`DataContext.fetch(...)`, but the legacy context surface still exists for
backward compatibility. Leaving both paths alive indefinitely keeps the public
API ambiguous.

**Validation at cleanup time:**

- targeted data-context routing tests
- relevant downstream compatibility checks
- `ruff check .`
- relevant `pytest` slices

### 2026-04-04

**Ticket:** `ALP-26`

**Title:**

`PIT cleanup: remove boolean strict compatibility for PIT ingestion`

**Trigger:**

- supported callers
- docs and examples

must no longer rely on:

- `strict=True`
- `strict=False`

for `PITAccessor.upsert_pit_observations(...)`

**Cleanup to perform:**

- remove boolean handling from PIT ingestion policy resolution
- require explicit string policies only:
  - `"error"`
  - `"warn"`
  - `"coerce"`
- update internal callers, docs, tests, and migration notes to use explicit
  strings only

**Why it exists:**

Boolean `strict` support is a backward-compatibility overload. The explicit
string policy is the clearer contract, and keeping both forms permanently adds
needless API ambiguity.

**Validation at cleanup time:**

- targeted PIT ingestion validation tests
- relevant downstream compatibility checks
- `ruff check .`
- relevant `pytest` slices

### 2026-04-04

**Ticket:** `ALP-27`

**Title:**

`PIT API: remove legacy get_*_ref compatibility helpers after ref-query migration`

**Trigger:**

- `nowcast-data`
- `positioning`
- other supported PIT consumers

must no longer rely on:

- `PITAccessor.get_snapshot_ref(...)`
- `PITAccessor.get_revision_timeline_ref(...)`

**Cleanup to perform:**

- remove or retire the legacy helper names from `alphaforge/pit/accessor.py`
- update PIT docs and examples to point only at:
  - `PITAccessor.snapshot_ref(...)`
  - `PITAccessor.revisions_ref(...)`
- remove compatibility-only tests that keep the old helper names alive

**Why it exists:**

`ALP-12` introduced the canonical typed ref-query surface but intentionally
left the older helper names in place to avoid an immediate downstream break.
Those helpers should not remain a parallel public API indefinitely.

**Validation at cleanup time:**

- targeted ref-query regression tests
- relevant downstream compatibility checks
- `ruff check .`
- relevant `pytest` slices
