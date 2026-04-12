# Core Platform Migration Guide

Use this guide when moving downstream code onto the canonical public surfaces
from the core platform roadmap.

## Downstream workflows in scope

The migration guidance is centered on the workflows that already depend on
Alphaforge:

- nowcast-style PIT semantics
- positioning-style operational source workflows
- volatility notebook dataset assembly

## Preferred public paths

Move new code toward these canonical surfaces:

| Area | Prefer this path |
| --- | --- |
| Temporal semantics | `from alphaforge.time import ...` |
| PIT ref queries | `RefSnapshotQuery`, `RefRevisionQuery`, `PITAccessor.snapshot_ref(...)`, `PITAccessor.revisions_ref(...)` |
| Adapter bootstrap | `DataContext.from_adapters(...)` |
| Common table loads | `ctx.load(...)` |
| Canonical routing | `ctx.fetch(...)`, `ctx.fetch_many(...)` |
| Research dataset assembly | `DatasetSpec`, `FeatureRequestGroup`, built-in templates |
| Source operations | `build_health_report(...)`, `discover_archive_fetches(...)`, `iter_yearly_archive_fetches(...)` |

## Compatibility surfaces still alive during migration

These surfaces are still available, but they should be treated as temporary
bridges rather than equal public directions:

- `alphaforge.pit.release_rules`
- `alphaforge.pit.missingness`
- `PITAccessor.get_snapshot_ref(...)`
- `PITAccessor.get_revision_timeline_ref(...)`
- `SourceAdapterPITCompat`
- `DataContext.sources`
- `DataContext.fetch_panel(...)`
- boolean `strict=True/False` PIT ingestion

Cleanup for those bridges is tracked in the post-migration backlog:

- `ALP-23`
- `ALP-24`
- `ALP-25`
- `ALP-26`
- `ALP-27`

## Recommended migration order

1. Move imports and typed semantics first.
2. Move PIT readers onto typed query objects and canonical execution methods.
3. Move data loading onto adapter-backed `DataContext` helpers.
4. Move notebook and dataset specs onto grouped requests and built-in
   templates where they fit.
5. Replace operational ad hoc logic with release-aware health reports and
   archive fetch plans.
6. Only remove a legacy surface after contract coverage and downstream callers
   are both clean.

## Required checks before deprecation or removal

Use the repo-local stability gates before tightening a contract:

```bash
python -m pytest tests/contracts
python -m benchmarks.pit
```

Also rerun the targeted subsystem tests for the touched layer and update the
repo-local migration notes:

- `doc/plan/migration_note.md`
- `doc/plan/active__post-migration-plan.md` when a temporary bridge is added or retired

## Documentation expectations

When a migration-sensitive ticket lands:

- update the canonical docs first
- leave legacy paths documented only as temporary compatibility notes
- record the downstream impact in `doc/plan/migration_note.md`
- keep the roadmap mirror in `doc/plan/draft__core-platform-roadmap.md` aligned with
  Linear state

That keeps downstream migrations tied to one canonical direction at a time.
