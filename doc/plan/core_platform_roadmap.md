# Alphaforge Core Platform Roadmap

**Status:** Draft for review

## Goal

Develop Alphaforge into a mathematically grounded platform for messy real-world
data, with a stable public API for:

- point-in-time data and revision-aware queries
- source access, caching, and archival ingestion
- research dataset assembly for tabular and PIT-driven workflows

This plan treats Alphaforge as a shared data platform, not as a full
strategy-orchestration framework.

## Why Now

The current downstream usage makes the next priorities clear:

- `nowcast-data` uses Alphaforge as a semantic PIT backend and stresses
  ref-period, revision, release, and as-of correctness.
- the volatility notebook workflow uses Alphaforge as a research assembly layer
  and stresses `DatasetSpec`, templates, and notebook-friendly ergonomics.
- `positioning` uses Alphaforge as the shared operational data layer and
  stresses public-web source robustness, PIT archival, and source health.

The public-web refactor is now largely complete, so the next roadmap should
consolidate the whole library rather than continuing one subsystem at a time.

## Design Thesis

Alphaforge exists because real-world data is messy. The library should impose
order on that mess by making the underlying mathematical structure explicit.

The central semantic object is not "a dataframe with dates". It is a value over
multiple axes:

- entity
- measure
- observation time or reference period
- availability time (`asof`)

For point-in-time data, we can think of the canonical relation as:

`x(entity, measure, observation, asof) -> value`

Everything else is derived from this:

- a snapshot fixes an `asof` cutoff and selects the latest admissible value
- a revision history fixes an observation key and varies `asof`
- a panel aligns multiple snapshots on an explicit evaluation grid
- a derived series is valid only if its transform is causal with respect to
  the chosen `asof`

That viewpoint should guide the public API, the internal constructs, and the
time-series operations.

## Semantic Laws

These are the invariants the implementation should preserve wherever possible.

### 1. Snapshot admissibility

A snapshot at cutoff `a` may only depend on observations whose availability is
`<= a`.

### 2. Revision identity

A revision history holds the observation key fixed and varies only the
availability axis. It is not a different time series; it is a different view of
the same semantic observation.

### 3. Explicit normalization

Any mapping between calendar dates and reference periods must be explicit and
frequency-aware. Alphaforge should not rely on silent date coercion where
ref-period semantics matter.

### 4. Causal transform safety

A PIT transform is valid only if, for every cutoff `a`, it can be evaluated from
inputs admissible at `a`. Non-causal transforms should be labeled accordingly.

### 5. Monotone information set

Moving `asof` forward may change values through new releases or revisions, but
it should never shrink the admissible information set.

## Mathematical Design Principles

### 1. Make semantic axes explicit

Never conflate:

- observation date
- reference period
- release date
- availability / `asof`
- evaluation grid
- trading calendar session label

If two concepts have different semantics, they should have different types,
fields, or APIs.

### 2. Prefer typed semantic objects over loose strings

Use explicit objects where semantics matter:

- ref periods
- release rules
- availability policies
- time grids
- missingness classification
- query intent for snapshots, revisions, and aligned panels

Strings and ad hoc dicts may remain at loader boundaries, but not as the main
semantic API.

### 3. Treat operations as algebra over typed data

Time-series operations should compose over clearly defined inputs:

- snapshot -> aligned series
- aligned series -> causal transform
- set of aligned series -> panel
- panel + policies -> dataset

Operations should declare their alignment and causality assumptions instead of
smuggling them through incidental pandas behavior.

### 4. Preserve causality by construction

For PIT workflows, every transform should either:

- be causal and safe under `asof`, or
- explicitly declare why it is not causal and where it is intended for
  evaluation-only use

Leakage should be a first-class failure mode, not an afterthought.

### 5. Separate semantic core from source plumbing

Source adapters, HTTP clients, caching, and archival concerns are essential, but
they should not define the semantics of PIT operations. The semantic core should
remain usable even when the underlying source families evolve.

### 6. Keep one canonical public path per layer

Alphaforge can keep compatibility shims, but it should not keep multiple equally
"official" abstractions indefinitely.

## Strategic Boundaries

Alphaforge should own:

- PIT semantics and transforms
- unified data access and caching contracts
- public-web and local archival source infrastructure
- dataset assembly and feature-template ergonomics
- source health, missingness, lineage, and leakage diagnostics

Alphaforge should not grow into:

- a full backtesting framework
- strategy-specific orchestration
- domain-specific modeling logic for every downstream project

## Platform Pillars

These are the enduring product pillars for Alphaforge. They are all important.
The implementation order later in this document reflects dependency structure,
not a belief that only the first few pillars matter.

### 1. Mathematical temporal semantics

Alphaforge should make time semantics explicit, typed, and defensible:

- ref periods
- release rules
- availability / `asof`
- calendars and evaluation grids
- missingness and causality

This is the formal substrate of the library.

### 2. PIT as the flagship capability

Revision-aware data is the strongest differentiator in the codebase. Alphaforge
should provide the best surface here:

- snapshots
- revisions
- aligned PIT panels
- causal transforms
- lineage and explainability

### 3. Unified source access and ingestion

The source layer should feel like one system rather than several overlapping
ones:

- one canonical fetch contract
- one routing story
- one compatibility story
- one archival ingestion story

### 4. Operational observability and data quality

Because the library deals with messy real-world data, operational discipline is
part of the product:

- source health
- release-aware staleness
- missingness classification
- leakage diagnostics
- lineage and provenance

### 5. Dataset algebra and research UX

Alphaforge should make research assembly elegant enough that notebooks and
experiments do not need to work around it constantly:

- `DatasetSpec`
- typed templates
- join and missingness policies
- notebook-friendly workflows
- reusable recipes for common feature families

### 6. Stability, compatibility, and performance

Downstream repos should be able to depend on Alphaforge without pinning to
implementation accidents:

- contract tests
- benchmarks
- migration guides
- deprecation policy
- release discipline

### 7. Public API ergonomics

The amount of code a downstream user needs to write to load something is a core
product signal. Alphaforge should not require repo-local helper layers for
routine tasks that ought to be first-class.

Common tasks should have short, task-shaped entry points for:

- loading a source table or panel
- loading a PIT snapshot
- loading a PIT revision history
- wiring a default local context
- building a small dataset from a concise spec

## Target Public API Shape

## Public API North Star

The public API should optimize for the common path, not merely expose the
internal architecture cleanly.

The roadmap should explicitly improve:

- time-to-first-frame: how many lines it takes to load a standard source table
- time-to-first-snapshot: how many lines it takes to load a PIT snapshot
- setup burden: how many objects a user must manually instantiate
- wrapper pressure: whether downstream repos still need bespoke loading facades

### Layer 1: Temporal Semantics

Build out `alphaforge.time` and adjacent PIT semantics around explicit objects:

- `RefPeriod` and ref-frequency-aware utilities
- `ReleaseRule`
- calendar-aware evaluation grids
- availability and missingness semantics

This layer is the mathematical vocabulary the rest of the library depends on.

### Layer 2: PIT Flagship API

Make the PIT layer the clearest public surface for revision-aware data:

- first-class snapshot queries
- first-class revision queries
- ref-period-aware snapshot and revision APIs
- batch snapshot / panel builders
- lineage and explainability for derived PIT series

Keep `PITAccessor` as the operational engine during migration, but move the
public surface toward typed query intent rather than ad hoc method expansion.

### Layer 3: Unified Data Access

Make `SourceAdapter` the canonical external data access contract.

The intended direction is:

- `SourceAdapter` = public, cache-aware source contract
- `DataContext.fetch(...)` = canonical routing entry point
- `DataSource` = legacy/raw-loader compatibility surface and internal bridge

This lets Alphaforge keep the current loader ecosystem without making the old
and new access models equally normative forever.

### Layer 4: Dataset Algebra

Make `DatasetSpec` the canonical research assembly API.

This layer should support:

- reusable feature and target requests
- typed template composition
- explicit join and missingness policies
- notebook-friendly single-entity workflows
- reusable recipes for common research patterns

### Layer 5: Operations and Observability

Keep source health, missingness, lineage, and leakage diagnostics as first-class
operational features rather than optional utilities.

## Downstream-Driven Success Criteria

The roadmap is successful when:

- `nowcast-data` no longer needs bespoke semantic wrappers for core ref-period,
  revision, and release-aware PIT behavior.
- `positioning` can build PIT panels and release-aware source health flows
  without repo-specific panel loops or duplicated health semantics.
- the volatility notebook workflow can express common feature-engineering paths
  with fewer manual joins and fewer bespoke notebook helpers.
- common downstream loading tasks require materially less boilerplate than they
  do today.
- public-web and archival workflows expose a clearer operational story for
  ingestion, health, and provenance rather than just a collection of loaders.
- Alphaforge publishes a clear migration path for compatibility layers and keeps
  regressions out with contract tests.

## Ordered Implementation Roadmap

The phases below are ordered primarily by dependency and architectural leverage.
They should not be read as "do nothing on later pillars until earlier phases are
fully complete". In practice, the roadmap should run as one foundation track and
several lighter parallel tracks.

## Execution Model

### Foundation Track

This is the dependency-critical path:

- temporal semantic core
- PIT public API consolidation
- data access contract unification

### Parallel Product Tracks

These should advance continuously in smaller slices while the foundation track
lands:

- dataset algebra and research UX
- operational data platform and observability
- public API ergonomics for common loading tasks

### Continuous Discipline

These should start early and continue throughout the roadmap:

- compatibility suites
- benchmarks
- migration docs
- deprecation and release policy

### Phase 1: Temporal Semantic Core

**Objective:** make time and release semantics explicit enough to support the
rest of the roadmap.

**Scope:**

- `alphaforge/time/`
- PIT release and missingness semantics
- typed ref-period and availability constructs

**Deliverables:**

- promote release rules into Alphaforge core
- add a unified missingness taxonomy and classifier
- standardize ref-period utilities and normalization rules
- document the semantic distinctions between observation time, ref period,
  release date, and `asof`

**Why first:** this is the foundation for both nowcast-style PIT correctness and
health / staleness generalization.

### Phase 2: PIT Public API Consolidation

**Objective:** make PIT the flagship user-facing surface.

**Scope:**

- `alphaforge/pit/`
- `alphaforge/time/ref_period.py`
- PIT docs and examples

**Deliverables:**

- first-class public APIs for snapshot, revision, and ref-period queries
- batch PIT panel builders for wide and long panel assembly
- lineage / explain APIs for derived PIT series
- contract tests covering causal transforms and as-of behavior

**Why second:** `nowcast-data` is already proving the importance of this layer,
and both `nowcast-data` and `positioning` are paying a performance tax through
manual snapshot loops.

### Phase 3: Data Access Contract Unification

**Objective:** reduce ambiguity between `DataSource`, `SourceAdapter`,
`PITDataSource`, and downstream compatibility wrappers.

**Scope:**

- `alphaforge/data/context.py`
- `alphaforge/data/adapter.py`
- `alphaforge/data/source.py`
- adapter compatibility shims

**Deliverables:**

- publish one canonical fetch path based on `SourceAdapter`
- narrow the role of `DataSource` to raw-loader / compatibility use
- document migration guidance for legacy source registration
- add contract tests for routing, dataset resolution, and cache-aware fetches

**Why third:** the unified access surface should be clarified before further
growth in dataset assembly or adapter families.

### Phase 4: Dataset Algebra and Research UX

**Objective:** make the research assembly layer feel deliberate rather than
partly-built.

**Scope:**

- `alphaforge/features/`
- dataset-spec docs and examples
- feature template catalog

**Deliverables:**

- tighten the `DatasetSpec` public contract
- expand built-in templates for calendar, event, rolling, and common market
  features
- add simpler single-entity and notebook-friendly patterns
- document recipe-style workflows that mirror the volatility notebooks

**Why fourth:** the volatility workflow is a lighter user of Alphaforge, but it
shows where the library should feel easier and more reusable.

### Phase 5: Operational Data Platform

**Objective:** complete the move from "loaders plus helpers" to a coherent
operational data layer.

**Scope:**

- source health
- archival helpers
- source metadata and diagnostics
- public-web operational workflows

**Deliverables:**

- integrate release-aware health policies where cadence alone is too weak
- provide better PIT archival helpers for recurring source ingestion
- standardize source metadata, health, and diagnostics surfaces
- consolidate docs for source authoring and operational usage

**Why fifth:** the public-web refactor is already done enough that the next
work should focus on operations and observability instead of more loader-family
cleanup.

### Phase 6: Stability, Compatibility, and Performance

**Objective:** make Alphaforge safe to depend on across downstream repos.

**Scope:**

- benchmarks
- contract tests
- migration and deprecation policy
- release discipline

**Deliverables:**

- benchmark PIT snapshot and panel workloads
- add compatibility suites modeled on current downstream usage
- publish deprecation windows for legacy surfaces
- ensure docs cover migration between the legacy and canonical APIs

**Why sixth:** performance and stability need to be enforced once the canonical
surfaces are clear enough to lock down.

## Proposed Implementation Slices

These slices define the intended implementation order. They are small enough to
become individual Linear tickets later.

1. Temporal semantics: adopt release rules and missingness taxonomy in
   Alphaforge core.
2. Temporal semantics: standardize ref-period normalization and typed helpers.
3. PIT API: add first-class ref-period snapshot and revision queries.
4. PIT performance: add batch snapshot and panel-building primitives.
5. PIT explainability: expose lineage and causal-transform diagnostics.
6. Data access: publish `SourceAdapter` as the canonical fetch contract.
7. Data access: narrow `DataSource` to a compatibility and raw-loader role.
8. Dataset API: tighten `DatasetSpec` semantics and template composition rules.
9. Public API ergonomics: reduce common loading boilerplate and setup friction.
10. Research UX: add notebook-ready templates and recipe documentation.
11. Operations: integrate release-aware health policies and archival helpers.
12. Compatibility: add downstream-inspired contract tests and benchmarks.
13. Documentation: publish migration guides and architecture references.

## First 90 Days

The first delivery window should stay foundation-first, but it should still
touch all major pillars in visible ways.

1. Formalize the semantic vocabulary for ref periods, release rules,
   availability, and missingness.
2. Turn the PIT layer into a clearer public API for ref-period snapshots,
   revisions, and aligned panel retrieval.
3. Add contract tests and benchmarks modeled on the current `nowcast-data` and
   `positioning` call patterns.
4. Publish the canonical direction for `SourceAdapter` versus `DataSource`
   before any larger adapter expansion.
5. Land at least one concrete research-UX slice in `DatasetSpec` or templates
   so the volatility workflow benefits early rather than waiting for a later
   phase.
6. Land at least one concrete operational slice for release-aware health or PIT
   archival helpers so the platform story is not purely semantic.
7. Land at least one concrete loading-ergonomics slice that reduces setup code
   for a common downstream task.

## 90-Day Workstreams

The first 90 days should run as parallel workstreams, not as a single-file
queue.

### Workstream A: Semantic foundation

- release rules
- missingness taxonomy
- ref-period normalization

### Workstream B: PIT flagship API

- ref-period snapshots
- revision queries
- aligned panel builders

### Workstream C: Research UX

- one or two high-value `DatasetSpec` / template improvements
- one notebook-shaped end-to-end example

### Workstream D: Loading ergonomics

- define the shortest supported path for common loading tasks
- remove one major source of manual context or adapter boilerplate

### Workstream E: Operations

- release-aware health policy integration
- one reusable archival helper improvement

### Workstream F: Stability

- downstream-inspired contract tests
- PIT performance benchmarks
- migration notes for canonical versus legacy surfaces

## Sequencing Constraints

- Finish the temporal semantic core before expanding PIT APIs that depend on it.
- Land PIT panel primitives before migrating downstream panel builders toward
  shared helpers.
- Clarify the canonical data access contract before broadening the adapter
  surface further.
- Improve loading ergonomics continuously rather than waiting for all
  architectural cleanup to finish.
- Treat dataset UX work as a second-order concern after PIT and access-contract
  consolidation.
- Use the public-web refactor as an input to Phase 5, not as the main roadmap.
- Do not remove legacy surfaces until contract tests and migration docs exist.

## Validation Plan

This is a planning-only document, so TDD does not apply yet. Implementation
tickets derived from this plan should validate against:

- targeted subsystem tests
- broader regression coverage for touched layers
- benchmark snapshots for PIT-heavy paths
- docs updates when public behavior changes

The highest-value contract coverage should model the three existing downstream
usage patterns:

- nowcast-style PIT semantics
- volatility notebook dataset assembly
- positioning-style operational source + health workflows

## Linear Ticket Mirror

These tables mirror the current Linear tickets for the core platform roadmap and
define the intended implementation order for subsequent agents.

Linear routing for this plan:

- Team: `ALP` (`alphaforge`)
- Project: `alphaforge`
- Umbrella issue: `ALP-9`
- Migration log: `doc/plan/migration_note.md`

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This plan file is the repo-local execution mirror for subsequent agents.
- `doc/plan/migration_note.md` is the running migration log for public-surface
  and downstream-impacting changes.
- Implement the earliest ticket in the ordered queue below whose status is not
  `Done` and whose hard prerequisites are already satisfied.
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
- Update `doc/plan/migration_note.md` for any ticket that changes the preferred
  public API, compatibility story, loading path, PIT semantics, or downstream
  migration actions.
- Update this table only after the ticket is closed in Linear.

Status mirror last synced: `2026-04-05`

### Ordered Epic Queue

| Ticket | Status |
| --- | --- |
| `ALP-9` Core platform: implement mathematically grounded public API and migration path | Done |

### Ordered Core Platform Queue

| Ticket | Status |
| --- | --- |
| `ALP-10` Temporal semantics: add release rules and missingness core | Done |
| `ALP-11` Temporal semantics: standardize typed ref-period utilities and normalization | Done |
| `ALP-12` PIT API: add first-class ref-period snapshot and revision queries | Done |
| `ALP-14` PIT performance: add batch snapshot and panel-building primitives | Done |
| `ALP-13` PIT explainability: expose lineage and causal diagnostics | Done |
| `ALP-15` Data access: canonicalize `SourceAdapter` fetch path | Done |
| `ALP-16` Data access: reduce `DataSource` to compatibility and raw-loader role | Done |
| `ALP-17` Dataset API: tighten `DatasetSpec` semantics and template composition | Done |
| `ALP-18` Public API ergonomics: reduce common loading boilerplate and setup friction | Done |
| `ALP-19` Research UX: add notebook-ready templates and recipe documentation | Done |
| `ALP-20` Operations: integrate release-aware health and archival helpers | Done |
| `ALP-21` Compatibility: add downstream contract suite and performance benchmarks | Done |
| `ALP-22` Docs: publish architecture and migration guides and maintain migration note | Done |

### Post-Migration Cleanup Queue

These tickets should not be picked up until their migration trigger conditions
are satisfied. The detailed cleanup backlog lives in
`doc/plan/post_migration_plan.md`.

| Ticket | Status |
| --- | --- |
| `ALP-23` Platform cleanup: remove temporal-semantics compatibility shims after migration | Backlog |
| `ALP-24` Platform cleanup: remove SourceAdapterPITCompat bridge after PIT adapter migration | Backlog |
| `ALP-25` Platform cleanup: remove legacy DataContext source access after adapter migration | Backlog |
| `ALP-26` PIT cleanup: remove boolean strict compatibility for PIT ingestion | Backlog |
| `ALP-27` PIT API: remove legacy get_*_ref compatibility helpers after ref-query migration | Backlog |

### Cross-Ticket Sequencing Constraints

Use the queue order above, but also respect these concrete handoff rules:

- Treat `ALP-9` as a tracking umbrella. Start implementation with `ALP-10`.
- Do not start `ALP-11` until `ALP-10` is landed.
- Do not start `ALP-12` or `ALP-14` until `ALP-11` is landed.
- Treat `ALP-12` and `ALP-14` as the first PIT-facing implementation queue;
  use `ALP-12` before `ALP-14` so the public query surface lands before the
  shared batch helpers.
- Use `ALP-13` after the first PIT public-surface slice is in place; it is an
  explainability pass, not the initial PIT contract.
- Use `ALP-15` before `ALP-16` and `ALP-18` so the canonical access path is
  clear before compatibility reduction and boilerplate reduction work.
- Use `ALP-17` before `ALP-19` so the dataset contract lands before the
  notebook-ready template and recipe pass.
- `ALP-20` depends on the semantic foundation from `ALP-10` but otherwise
  should advance alongside the main queue once that dependency is satisfied.
- `ALP-21` should begin after the first meaningful public-surface slices are in
  place; it is a continuous discipline ticket, but it should not block the
  roadmap from starting.
- `ALP-22` starts immediately and remains active throughout the umbrella. Do
  not treat it as a substitute for implementation tickets.

## Agent Pickup Directive

If this roadmap is handed to a coding agent for implementation planning:

1. start at the top of the epic queue
2. within the active queue, pick the first non-`Done` ticket whose hard
   prerequisites are satisfied
3. implement only that ticket's scoped outcome
4. update `doc/plan/migration_note.md` whenever the ticket changes downstream
   behavior or migration expectations
5. close the ticket in Linear first
6. then update the mirrored status row in this plan before moving on

## Non-Goals

- turning Alphaforge into a full backtesting or strategy package
- hiding all source-specific behavior behind a generic DSL
- forcing deep inheritance on unrelated data-source families
- introducing abstract machinery without at least one clear downstream user

## Immediate Next Step

The first implementation umbrella should cover Phases 1 and 2 together:

- formalize the semantic vocabulary for time, release, and availability
- then use that vocabulary to make PIT the clearest public API in the library

That is the highest-leverage path because it addresses the heaviest downstream
pressure while setting the mathematical foundation for the rest of the platform.
