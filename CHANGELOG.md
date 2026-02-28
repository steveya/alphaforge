# Changelog

## Unreleased

- Added canonical PIT table and PIT accessor for snapshot and revision timeline queries.
- Added reference period keys and PIT ref-based queries for snapshot ranges and revision timelines.
- Added PIT transform engine (`apply_transform`, `list_transforms`) with lineage/run tables.
- Added PIT safety toolkit (`ReleaseLagPolicy`, `effective_asof`, `pit_leakage_report`).
- Added `PITDataSource` with `pit.snapshot` and `pit.observations` contracts.
- Added PIT helper tasks for vintage selection, revision analytics, staleness, and YoY/QoQ transforms.
- Added experimental `revision_path` transforms for `diff`, `lag`, `rolling`, and `expanding`.
