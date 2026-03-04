# PIT Migration Notes

This guide tracks PIT contract versions and required migration actions for breaking validation or API behavior changes.

## Template

- **Version:** semantic version string
- **Date:** YYYY-MM-DD
- **Change Type:** breaking | non-breaking
- **Summary:** short statement of what changed
- **Required Actions:** concrete migration steps for downstream projects

## Entries

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
