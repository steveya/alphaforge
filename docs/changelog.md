# Changelog

This page mirrors release notes from [`CHANGELOG.md`](https://github.com/steveya/alphaforge/blob/main/CHANGELOG.md).

## Unreleased

- Added PIT cross-series `op="binary"` transforms (`add`/`sub`/`mul`/`div`) with deterministic lineage and causality-preserving execution.
- Added cross-series transform lineage payloads with per-input source as-of timestamps and multi-input transform metadata.
- Expanded PIT task utilities with `revision_event_stream(...)` and `revision_volatility(...)`.
- Enhanced staleness helper output (`forward_fill_with_staleness`) with `source_obs_date` and `age`.
- Added CI-verified runnable PIT docs examples (`tests/test_docs_pit_examples.py`) and a dedicated `docs-examples` CI job.
- Added PR guard script (`scripts/check_pit_docs_sync.py`) and CI job to enforce PIT docs + changelog updates when PIT public interfaces change.
- Added PIT typed exception hierarchy for deterministic contract/error handling.
- Added strict PIT ingestion validation API (`PITValidationReport`, `validate_pit_observations`) and made strict validation the default for `upsert_pit_observations`.
- Tightened transform parameter contracts with per-operator key validation and canonical normalization.
- Added `PITAccessor.preview_transform(...)` and dict-based transform spec compatibility.
- Added explicit `allow_experimental=True` gating for `axis=\"revision_path\"`.
- Updated transform engine contract so `auto` prefers DuckDB for supported specs and falls back deterministically to Python when configured.
- Expanded `PITTransformResult` with requested engine and fallback metadata.
- Tightened `PITDataSource` query contract (`pit.snapshot` requires `asof`; unsupported vintage modes now fail explicitly).
- Added PIT query helpers: `PITDataSource.snapshot_query(...)` and `PITDataSource.observations_query(...)`.
- Added canonical PIT table and PIT accessor for snapshot and revision timeline queries.
- Added reference period keys and PIT ref-based queries for snapshot ranges and revision timelines.
- Added PIT transform engine (`apply_transform`, `list_transforms`) with lineage/run tables.
- Added PIT safety toolkit (`ReleaseLagPolicy`, `effective_asof`, `pit_leakage_report`).
- Added `PITDataSource` with `pit.snapshot` and `pit.observations` contracts.
- Added PIT helper tasks for vintage selection, revision analytics, staleness, and YoY/QoQ transforms.
- Added experimental `revision_path` transforms for `diff`, `lag`, `rolling`, and `expanding`.
