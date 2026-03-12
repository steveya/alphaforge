# PIT Migration Notes

This guide tracks PIT contract versions and required migration actions for breaking validation or API behavior changes.

## Template

- **Version:** semantic version string
- **Date:** YYYY-MM-DD
- **Change Type:** breaking | non-breaking
- **Summary:** short statement of what changed
- **Required Actions:** concrete migration steps for downstream projects

## Entries

### Version 2.1.0

- **Date:** 2026-03-10
- **Change Type:** non-breaking
- **Summary:** Added adjusted PIT `splice`, explicit PIT fold generators, and filtered vs retrospective snapshot-tape contracts.
- **Required Actions:**
  1. Existing transform, pipeline, and expression-graph code remains valid without changes.
  2. Use `splice` instead of hand-rolled continuation logic when ratio/additive adjusted PIT continuation is required.
  3. Prefer `iter_walk_forward_folds(...)`, `iter_purged_kfold_folds(...)`, and `build_snapshot_tape(...)` for training/validation/filtering loops so as-of semantics are explicit.

### Version 2.0.0

- **Date:** 2026-02-28
- **Change Type:** non-breaking
- **Summary:** Added ingestion policy modes (`strict="error|warn|coerce"`), release selection helpers, expression graph APIs, union-vintage utility, and snapshot panel builder.
- **Required Actions:**
  1. Existing `strict=True/False` calls remain valid; `False` now maps to warn semantics explicitly.
  2. Downstream nowcast release-rank logic can migrate to `list_release_stream(...)` and `resolve_release(...)`.
  3. Prefer expression graph APIs for multi-input PIT bridge features to standardize lineage and as-of alignment.


### Version 2.0.1

- **Date:** 2026-03-04
- **Change Type:** non-breaking
- **Summary:** Fixed stale `type: ignore` comments and TypedDict narrowing in `pit/models.py` and `pit/accessor.py`. No runtime behavior change.
- **Required Actions:** None.

### Version 2.0.2

- **Date:** 2026-03-12
- **Change Type:** non-breaking
- **Summary:** Code-quality fixes only — import block sorting, removal of unused imports, and mypy type-narrowing assertion in `iter_walk_forward_folds`. No public API or runtime behavior change.
- **Required Actions:** None.
