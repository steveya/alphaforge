# Alphaforge — Multi-Agent Collaboration Guide

## Repository Overview

Alphaforge is a point-in-time data and feature engineering library for
systematic research.

**Language:** Python 3.10+
**Core dependencies:** pandas, duckdb, pyarrow, PyYAML
**Python:** `/Users/steveyang/miniforge3/bin/python`
**Tests:** `pytest`
**Lint:** `ruff check .`
**Type check:** `mypy alphaforge`
**Docs:** `mkdocs build --strict`

Prefer the miniforge interpreter above unless a repo-local virtualenv exists and
has already been adopted for the current task.

## Agent Roles

### Role: Data Source Developer
**Scope:** `alphaforge/data/public_web/`, `alphaforge/data/sources/`,
`alphaforge/data/transforms/`, `alphaforge/data/registries/`
**Task:** Build, refactor, or extend public-web loaders, source adapters,
registry-backed metadata, and source-specific PIT transforms.
**Rules:**
- Read the source module, its matching test file, and shared helpers before
  editing.
- Preserve table names, schema contracts, entity-id semantics, sorting, column
  projection, and `asof_utc` behavior unless the ticket explicitly changes the
  contract.
- Keep shared abstractions shallow until at least three sources clearly benefit.
- Migrate only a small family of loaders at a time.
- Update targeted source tests in `tests/public_web/` and broader adapter or
  regression tests in `tests/` when routing or contracts change.

### Role: Test & Validation Agent
**Scope:** `tests/`, `tests/public_web/`
**Task:** Add or update targeted tests, broader regression coverage, and review
gates for changed behavior.
**Rules:**
- Prefer the smallest targeted failing test first, then broaden to subsystem
  regression coverage.
- Review code and tests together; missing interaction coverage is a finding, not
  a note for later.
- When compatibility-only behavior must remain, keep the compatibility boundary
  explicit in tests instead of mixing it into ordinary happy-path coverage.
- Run the narrowest useful validation first, then the broader commands required
  by the ticket or changed subsystem.

### Role: Documentation Agent
**Scope:** `docs/api/`, `docs/getting-started/`, `docs/guides/`
**Task:** Keep API reference, onboarding, workflow, and conceptual docs aligned
with shipped behavior.
**Rules:**
- Update docs after implementation and tests stabilize.
- Treat behavior, API, workflow, source-coverage, and validation changes as
  documentation work unless the ticket is explicitly doc-free.
- Do not document unsupported source behavior or time semantics that code and
  tests do not prove.

### Role: Planning Agent
**Scope:** `doc/plan/`
**Task:** Maintain mirrored ticket tables, implementation order, and review or
cleanup backlog notes that future agents rely on.
**Rules:**
- Keep mirrored plan rows aligned with Linear, not ahead of it.
- Keep ticket tables sorted in implementation order.
- When a shared abstraction or program plan changes scope, update the relevant
  plan doc in the same slice.

## Module Ownership

| Path | Owner | Notes |
|------|-------|-------|
| `alphaforge/data/public_web/` | Data Source Developer | Public web source loaders, parsing, HTTP helpers |
| `alphaforge/data/sources/` | Data Source Developer | Unified source adapters and cache-aware wrappers |
| `alphaforge/data/transforms/` | Data Source Developer | Source-specific PIT transforms |
| `alphaforge/data/registries/` | Data Source Developer | Registry-backed source metadata |
| `tests/public_web/` | Test & Validation Agent | Public web loader coverage |
| `tests/` | Test & Validation Agent | Broader regression coverage |
| `docs/api/` | Documentation Agent | API reference updates |
| `docs/getting-started/` | Documentation Agent | Onboarding and quickstart guidance |
| `docs/guides/` | Documentation Agent | Workflow and conceptual guides |
| `doc/plan/` | Planning Agent | Repo-local implementation plans and mirrored ticket tables |

## Coordination Protocol

### When changing public-web loaders

1. Read the source module, its matching test file, and any shared helper modules
   it relies on.
2. Preserve table names, schema contracts, and entity-id semantics unless the
   ticket explicitly calls for contract changes.
3. Update or add targeted tests in `tests/public_web/`.
4. If the change affects higher-level routing, also update the relevant adapter
   tests in `tests/`.
5. Update docs if behavior, supported sources, or developer workflow changes.

### When adding or changing shared abstractions

1. Keep the abstraction shallow until at least three sources clearly benefit.
2. Migrate only a small family of loaders at a time.
3. Verify that empty-frame behavior, column projection, sorting, and
   `asof_utc` semantics stay stable.
4. Update the relevant plan doc in `doc/plan/` if the abstraction changes
   implementation order or scope.

### When validation or compatibility boundaries change

1. Review code and tests together.
2. Add or update targeted tests before broadening the regression scope.
3. Keep compatibility-only tests clearly separated from ordinary API-surface
   coverage.
4. Update docs or plan notes when the validation strategy or migration boundary
   changes.

## Repo Tracking Configuration

This repo inherits the shared Linear, planning, implementation, and review
workflow from [/Users/steveyang/Projects/steveya/AGENTS.md](/Users/steveyang/Projects/steveya/AGENTS.md).
If `alphaforge` is opened directly instead of from the workspace root, read the
workspace file before starting Linear-tracked work.

- Linear project: `alphaforge`
- Linear team: `alphaforge`
- Ticket prefix: `ALP`
- Queue-driving plan docs should live in `doc/plan/` and use the shared
  `draft__`, `active__`, or `done__` filename prefixes.
- Supporting notes that are not themselves the active queue may keep
  descriptive names, for example `migration_note.md`.
- Official docs roots:
  - `docs/api/`
  - `docs/getting-started/`
  - `docs/guides/`
- Review program path: use `doc/plan/active__code-review-program.md` when one
  exists; otherwise treat the user-directed scope or current diff as the review
  slice.
- Repo-specific naming should keep the temporal or PIT domain explicit when the
  work touches temporal semantics, PIT APIs, dataset contracts, compatibility
  shims, or public-web source families.

## Anti-Hallucination Rules

1. Never invent table names, entity ids, registry keys, or source names. Verify
   from code or tests.
2. Never claim a public-web source supports a format or time semantics unless
   the loader or tests prove it.
3. Never silently broaden a shared abstraction across unrelated source
   families.
4. Check file targets before patching. Resolve paths from repo root and verify
   with `rg --files` or `git status`.
5. Check docs targets before updating them. Use only existing doc trees unless
   the ticket explicitly introduces a new one.

## Key Files to Read First

1. `README.md`
2. `pyproject.toml`
3. `doc/plan/`
4. `alphaforge/data/public_web/registry.py`
5. `alphaforge/data/public_web/http.py`
6. `alphaforge/data/public_web/parsing.py`
7. `alphaforge/data/public_web/utils.py`
8. `tests/public_web/test_source_test_mapping.py`
