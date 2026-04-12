# CFTC Commodity CoT Loader Completion Plan

## Objective

Audit and complete Alphaforge's public-web surface for CFTC commodity
Commitments of Traders data.

This plan is intentionally framed as a completion plan rather than a greenfield
loader build. The repo already ships `CFTCDisaggregatedCoTSource` on
`cftc.cot.disagg`, so the first job is to confirm whether that existing surface
already satisfies the intended commodity CoT requirement or whether the real
gap is narrower.

Status mirror last synced: `2026-04-12`

Current status:

- the umbrella workstream exists in Linear as
  [ALP-39](https://linear.app/quant-macro/issue/ALP-39/public-web-cftc-commodity-cot-loader-completion)
- `cftc.cot.disagg` already exists in code, tests, and docs, so the current
  risk is scope drift or contract ambiguity rather than a missing file
- the audit has already identified one concrete mismatch: the public quickstart
  example was using adapter-style `value` / bare-entity semantics against the
  raw public-web loader contract
- `ALP-40` is done: the audit confirmed there was no missing loader
- `ALP-41` is canceled: no code-hardening slice remained after the audit
- `ALP-42` is done: docs and validation now match the shipped raw-loader
  contract
- the workstream is complete and should move to `done__` status in the plan
  mirror

## Why This Plan Exists

The current request is phrased as if Alphaforge still needs a commodity CoT
loader. The repo state does not match that framing:

- `alphaforge/data/public_web/cftc_cot.py` already defines
  `CFTCDisaggregatedCoTSource`
- `alphaforge/data/sources/cftc.py` already exposes `cot.disagg`
- `tests/public_web/test_cftc_cot.py` already covers disaggregated commodity
  fixtures, entity ids, and archive URL selection
- public docs already mention `cftc.cot.disagg`

That means the correct planning move is to audit the intended commodity scope
first, then either:

- confirm the current loader is already the right implementation and close the
  request with docs and validation evidence, or
- land the smallest reviewable hardening slices needed to close a real contract
  gap

## Repo-Grounded Current State

### Implemented surfaces

- Public-web loader:
  `alphaforge/data/public_web/cftc_cot.py`
- Adapter routing:
  `alphaforge/data/sources/cftc.py`
- Targeted public-web coverage:
  `tests/public_web/test_cftc_cot.py`
- Adapter and PIT coverage:
  `tests/test_cftc_dtcc_adapter.py`
- Public docs:
  `docs/api/public-web.md`
  `docs/getting-started/quickstart-public-web.md`
  `docs/guides/public-web-source-authoring.md`

### Known ambiguity this plan resolves

The existing commodity surface is the disaggregated futures report. The user
request may instead mean:

- validate that `cftc.cot.disagg` is the intended commodity CoT variant
- add missing metrics, entity mapping, or archive behavior to that existing
  loader
- align docs and examples so downstream users know which commodity CoT variant
  Alphaforge actually supports

This plan does not assume the answer in advance. `ALP-40` is responsible for
making that explicit from the current code and tests.

## Scope

- `alphaforge/data/public_web/cftc_cot.py`
- `alphaforge/data/sources/cftc.py`
- `tests/public_web/test_cftc_cot.py`
- `tests/test_cftc_dtcc_adapter.py` when adapter parity moves
- `docs/api/public-web.md`
- `docs/getting-started/quickstart-public-web.md`
- `docs/guides/public-web-source-authoring.md`

## Non-goals

- adding unrelated CFTC datasets such as swaps, supplemental, or index-trader
  reports
- redesigning the shared archive-loader stack unless a concrete commodity CoT
  bug demands a narrow fix
- changing PIT transform semantics unless the audit proves the commodity CoT
  contract currently depends on the wrong transform boundary

## Ordered Ticket Mirror

Rules for coding agents:

- Linear is the source of truth for ticket scope and status.
- This plan is the repo-local queue mirror for subsequent agents.
- Always start with `ALP-40`.
- If `ALP-40` concludes that the current implementation already satisfies the
  intended commodity scope, rewrite or close `ALP-41` before writing
  unnecessary code.
- Do not mark any row `Done` here before the corresponding Linear issue is
  actually closed.

### Ordered Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| [ALP-40](https://linear.app/quant-macro/issue/ALP-40/public-web-audit-existing-cftc-commodity-cot-loader-contract) | Audit the existing `cftc.cot.disagg` contract and state the real remaining gap | Done |
| [ALP-41](https://linear.app/quant-macro/issue/ALP-41/public-web-harden-cftc-commodity-cot-loader-gaps) | Land the smallest code changes needed after the audit | Canceled |
| [ALP-42](https://linear.app/quant-macro/issue/ALP-42/public-web-document-and-validate-cftc-commodity-cot-surface) | Align docs, validation, and closeout with the final supported surface | Done |

## Validation

Minimum expected validation by slice:

- `ALP-40`
  - static audit of loader, adapter, tests, and docs
- `ALP-41`
  - targeted `tests/public_web/test_cftc_cot.py`
  - targeted adapter slices if dataset routing, source naming, or series-key
    behavior changes
- `ALP-42`
  - rerun relevant targeted tests after the final code state
  - run docs validation if doc content changes materially

## Closeout Criteria

This plan is complete only when:

- the intended commodity CoT variant is explicitly identified
- the existing loader is either confirmed sufficient or hardened through a
  narrow implementation slice
- docs and examples reflect the actual supported commodity CoT surface
- the Linear handoff notes record the validation and any residual caveats

Those conditions are now met. Keep this file as the historical mirror under
`done__cftc-commodity-cot-loader.md`.
