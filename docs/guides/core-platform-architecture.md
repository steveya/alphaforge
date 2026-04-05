# Core Platform Architecture

This guide describes the canonical layer boundaries established by the core
platform roadmap.

## Layer 1: Temporal semantics

`alphaforge.time` is the canonical home for shared time semantics:

- `RefPeriod`
- `ReleaseRule`
- release-rule helpers such as `FixedLagMonths`
- missingness classification

Use this layer when code needs to talk about reference periods, release timing,
or missingness explicitly. Do not treat legacy PIT import paths as co-equal
public directions.

## Layer 2: PIT flagship API

`alphaforge.pit` is the flagship analytical layer built on top of the temporal
core.

The canonical ref-period query path is:

- `RefSnapshotQuery`
- `RefRevisionQuery`
- `PITAccessor.snapshot_ref(...)`
- `PITAccessor.revisions_ref(...)`
- `PITAccessor.build_snapshot_panel_long(...)`

Lineage and causality diagnostics also live here, so PIT transforms,
explainability, and revision-aware panels share one semantic surface.

## Layer 3: Source access and routing

The canonical data-loading path is adapter-first:

- `SourceAdapter`
- `DataContext.from_adapters(...)`
- `DataContext.fetch(...)`
- `DataContext.fetch_many(...)`
- `DataContext.load(...)`

`DataSource` and `DataContext.sources` remain for raw-loader and migration
scenarios, but they are not the long-term public direction.

## Layer 4: Dataset algebra and research UX

Research assembly sits in `alphaforge.features` and `alphaforge.features.dataset_spec`.

The intended composition model is:

- `DatasetSpec`
- `FeatureRequestGroup`
- built-in templates such as `LagReturnsTemplate`
  and `RollingVolatilityTemplate`
- `build_dataset(...)`

This layer should express joins, missingness policy, grouped feature families,
and notebook-friendly recipes without downstream helper scaffolding.

## Layer 5: Operations and observability

Operational data workflows build on the same canonical surfaces:

- release-aware source health via `SourceHealthPolicy`
  and `build_health_report(...)`
- deterministic archive planning via `ArchiveFetchPlanEntry`,
  `discover_archive_fetches(...)`, and `iter_yearly_archive_fetches(...)`

These APIs are intended to keep health monitoring and archival ingestion
explicit, typed, and reusable across public-web sources.

## Layer 6: Stability and migration discipline

Stability is enforced with two repo-local gates:

- `tests/contracts`
- `python -m benchmarks.pit`

Use them alongside targeted subsystem tests whenever work changes a canonical
surface or a temporary compatibility path.

## Canonical versus compatibility surfaces

Use the canonical side for new work:

| Area | Canonical surface | Compatibility-only or legacy surface |
| --- | --- | --- |
| Temporal semantics | `alphaforge.time.*` | `alphaforge.pit.release_rules`, `alphaforge.pit.missingness` |
| PIT ref queries | `snapshot_ref(...)`, `revisions_ref(...)` | `get_snapshot_ref(...)`, `get_revision_timeline_ref(...)` |
| Data loading | adapter-backed `DataContext.fetch/load/fetch_many` | `DataContext.sources`, `fetch_panel(...)`, raw `DataSource` routing |
| PIT adapter bridge | source-adapter path | `SourceAdapterPITCompat` |
| PIT ingestion strictness | `"error"`, `"warn"`, `"coerce"` | boolean `strict=True/False` |

Those compatibility surfaces remain only to support downstream migration.
Their removal backlog is tracked in `doc/plan/post_migration_plan.md`.
