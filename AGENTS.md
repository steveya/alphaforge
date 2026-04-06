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

## Linear Issue Writing Spec

Linear is the shared work ledger for cross-agent work. Issues should be
specific, searchable, dependency-aware, and tied to an observable outcome.
Treat an issue as a short engineering spec, not as a note or a chat summary.

### When to create or update an issue

- Create a Linear issue when work must survive beyond the current chat, spans
  more than one file or module, introduces a blocker, or needs durable tracking.
- Update an existing issue instead of creating a duplicate when the scope is
  the same.
- Split a ticket when it contains more than one independent reviewable outcome.
  Use an umbrella issue for the broad objective and child issues for delivery
  slices.
- Do not mirror every trivial local task into Linear. Use Linear for durable
  planning, blockers, coordination, and user-visible work.

### Title and naming

- Use outcome-first titles of the form `<Area>: <specific result>`.
- Keep titles short, concrete, and searchable.
- Put the domain noun in the title, not just the implementation verb.
- If the work concerns temporal semantics, PIT APIs, dataset contracts,
  compatibility shims, or public-web source families, name that domain
  explicitly in the title.
- Avoid vague titles such as `Cleanup`, `Refactor`, or `Improve module`
  unless paired with the exact target.
- Good examples:
  - `Public web: extract shared source finalization helpers`
  - `Registry APIs: add base class for entity-driven public sources`
  - `Archive loaders: unify historical-batch URL selection`

### Issue body shape

Use a compact spec structure:

- Objective: what should exist when the issue is done.
- Why now: why this matters now.
- Scope: the exact modules, files, or docs in scope.
- Non-goals: what is explicitly out of scope.
- Dependencies: hard blockers with issue IDs.
- Acceptance criteria: observable conditions that define success.
- Validation: tests, lint, docs, or review gates.
- Follow-on work: separate issues for future slices, if needed.

### Dependency rules

- Use parent/child relationships for umbrella work and implementation slices.
- Use `blockedBy` and `blocks` only for hard prerequisites.
- Use `relatedTo` for adjacent work that does not prevent completion.
- Keep blocker chains shallow.
- If the blocker does not yet exist, create it first or state the missing
  prerequisite explicitly.
- Do not block on anticipated future reuse alone; keep that as a scoped note
  unless the dependency is already real.

### Priority rules

- Priority 1 / Urgent: broken build, release blocker, or active outage.
- Priority 2 / High: foundational platform work or an item that unlocks
  multiple other tickets.
- Priority 3 / Normal: planned implementation slices and most feature work.
- Priority 4 / Low: docs-only work, cleanup, exploratory refactors.
- Default to Priority 3 unless there is a concrete reason to raise it.

### Blocker handling

- If a task is blocked, say so explicitly in the issue and when communicating
  with the user.
- State the blocking issue ID(s), the missing prerequisite, and the next
  unblock step.
- Never present a blocked issue as complete.
- If the user asks to complete work but a Linear blocker remains, surface the
  blocker before claiming success.

### Done criteria

- Mark an issue Done only when the implementation slice is landed, validation
  passes, and acceptance criteria are satisfied.
- If behavior, APIs, or supported-source coverage changed, update the relevant
  docs:
  - `docs/api/` for API and reference behavior
  - `docs/getting-started/` for onboarding or examples
  - `docs/guides/` for workflows and conceptual docs
  - `doc/plan/` for mirrored ticket tables and implementation plans
- Close the issue with a short note summarizing the result, validation, docs
  changes, and any follow-on issue IDs.
- If useful work remains, split it into follow-on issues instead of leaving the
  original issue ambiguous.

### Umbrella closeout

- Treat an umbrella issue as a maintenance checkpoint, not just a delivery
  milestone.
- Before closing an umbrella, do the cleanup pass, documentation maintenance,
  and mirrored-plan updates that the completed slices imply.
- If an umbrella is intentionally doc-free, record that explicitly in the
  closeout note and explain why no docs changed.

### Engineering plan quality

- Every implementation issue should include a concrete plan with small phases.
- Prefer stable scaffolds plus surgical deltas over whole-module rewrites.
- If the plan cannot be explained as a few reviewable phases, the issue is too
  large and should be split.

## Ticket Implementation Workflow

All coding agents working from Linear must follow this workflow for every
implementation ticket unless the user explicitly overrides it.

### Required execution order

1. Review upstream context before coding.
2. Announce the current ticket number and its plain-English goal on screen.
3. Implement the current ticket with test-driven development.
4. Update the relevant docs after the implementation and tests pass.
5. Leave a handoff note in Linear describing what changed and any caveats.
6. Mark the ticket `Done`, then update the mirrored ticket table in the
   relevant plan doc.

### Step 1: Review upstream context

Before writing code, the implementer must:

- read the current ticket body in full
- read all hard-blocking upstream tickets and their completion notes
- read recent comments on the parent issue when the parent is an active
  umbrella ticket
- inspect the referenced plan docs and the current code paths in scope
- identify the exact files, tests, and docs that are likely to change

If an upstream ticket is not done, do not start implementation unless the user
explicitly approves working around the blocker.

### Step 2: Announce the current ticket on screen

Before coding, print the ticket number and a short plain-English explanation
of what the ticket aims to do in the current terminal/chat session.

Minimum expectation:

- include the Linear ticket id, for example `ALP-123`
- explain the ticket goal in one or two plain-English sentences
- do this after reading upstream tickets and before writing code

### Step 3: Implement with TDD

For code-changing tickets:

- start by adding or updating the tests that define the target behavior
- run the tests and confirm they fail for the expected reason
- implement the smallest coherent code change that makes the tests pass
- rerun the targeted tests, then rerun the broader validation required by the
  ticket or module owner rules

Minimum expectation:

- targeted tests for the changed behavior
- broader regression coverage for the touched subsystem when practical

For documentation-only or planning-only tickets, state explicitly in the ticket
that TDD does not apply.

### Step 4: Update docs after tests pass

When behavior, APIs, runtime flow, source coverage, governance, validation, or
developer workflow changes, update the relevant docs after the implementation is
stable:

- `docs/api/` for API and source reference behavior
- `docs/getting-started/` for quickstart and setup
- `docs/guides/` for workflows and design guidance
- `doc/plan/` for mirrored ticket tables and implementation plans

Doc updates are part of completing the ticket, not optional follow-up work,
unless the ticket is explicitly scoped as doc-free.

### Step 5: Leave a Linear handoff note

Before marking the ticket done, add a Linear comment with:

- what was implemented
- which tests were added or updated
- which test commands were run and whether they passed
- which docs were updated
- any caveats, deferred work, follow-on risks, or compatibility notes

If the ticket is blocked or only partially complete, leave the same note but do
not mark it done.

### Step 6: Mark done and update the mirrored plan table

Linear is the source of truth for ticket state. The plan tables in `doc/plan/`
are the repo-local mirror for subsequent coding agents.

Mark the ticket `Done` only when:

- the code is landed
- the agreed validation passed
- the relevant docs are updated or the ticket explicitly records why no docs
  changed
- the Linear handoff note is written

After the ticket is moved to `Done` in Linear:

- update the corresponding plan-table row in the relevant plan doc
- keep the table sorted in implementation order
- skip tickets already marked `Done` when selecting the next ticket
- do not mark the plan-table row `Done` before the Linear ticket is actually
  closed

### Recommended Linear closeout template

Use this structure for the final implementation note:

- Implemented:
- Tests:
- Docs:
- Caveats:
- Follow-ons:

## Code Review Workflow

When the task is a code review rather than a feature implementation, use the
repo-wide review program under `doc/plan/` when one exists and the linked
Linear review workstream for ticket state. If Alphaforge does not yet have a
dedicated review program doc or review-ticket queue, treat the user-directed
scope or the current branch diff as the review slice and follow the same
dossier, severity, and closeout standards.

### Review ticket selection

- Linear is the source of truth for review-ticket state when review tickets
  exist.
- Use the mirrored queue in `doc/plan/` when a dedicated review-program doc is
  present.
- Pick the earliest review ticket in the ordered queue whose status is not
  `Done`, unless the user explicitly redirects to a different slice.
- If no dedicated review queue exists yet, review the current branch diff or the
  user-directed scope instead of inventing tickets.
- Tickets in the same wave may run in parallel only when their write scopes do
  not conflict.

### Review scope and posture

- Treat review tickets as review-and-fix slices, not as read-only audits.
- Review code and tests together.
- Check local, regional, and global behavior:
  - Local: single function, class, or module behavior.
  - Regional: bounded interactions across modules, adapters, registries,
    transforms, or docs.
  - Global: user-visible or workflow-visible behavior across major layers.
- For important behavior, missing regional or global interaction coverage is a
  review finding, not a note for later.
- Keep compatibility-only coverage isolated from the ordinary API-surface
  suites.
- Use the available test strata explicitly during review and remediation:
  - targeted source or module tests for local behavior
  - adapter, contract, or regression tests for bounded interactions
  - broader workflow or docs validation when the slice affects user-visible
    behavior

### Required review workflow

1. Read the review ticket or selected scope, the mirrored plan section if one
   exists, and any upstream blockers.
2. Build a review dossier:
   - files in scope
   - tests in scope
   - contracts being defended
   - inbound and outbound module interactions
   - current local / regional / global coverage map
   - relevant docs, plan notes, and compatibility or migration notes
3. Announce the ticket number or review slice and the plain-English review goal
   before editing.
4. Perform the static review of code and tests.
5. Run the targeted tests for the slice and the broader regression required by
   the ticket when practical.
6. Land low-risk in-scope fixes, test additions, doc updates, or plan updates
   inside the same ticket when the user asked for review-and-fix work.
7. Split larger remediations into follow-on Linear issues instead of letting
   the review ticket sprawl.

### Findings and severity

- Findings are the primary output of a review ticket.
- Use this severity rubric:
  - `P0`: wrong result, silent corruption, or broken governance on a critical
    path
  - `P1`: high-confidence correctness, runtime, or contract bug
  - `P2`: meaningful maintainability, test, or observability gap with real risk
  - `P3`: lower-risk cleanup or consistency gap

### Review closeout requirements

Before marking a review ticket `Done`:

- leave a Linear handoff note with:
  - Reviewed:
  - Findings:
  - Tests checked:
  - Coverage assessment:
  - Docs / plan mismatches:
  - Compatibility or shim removal candidates:
  - Follow-on tickets:
  - Disposition:
- update the mirrored ticket row in the review-program doc when one exists
- do not mark the ticket `Done` if a hard blocker remains unresolved

### Code review deliverables

Every completed review ticket should leave behind:

- prioritized findings with file references
- a local / regional / global coverage assessment
- missing-test and misleading-test notes
- docs and plan mismatches
- compatibility or shim-removal notes where relevant
- follow-on issues for out-of-scope or larger remediations

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
