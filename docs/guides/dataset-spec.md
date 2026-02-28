# Dataset Spec Guide

`DatasetSpec` is the declarative contract for reproducible dataset builds.

## Main components

- `UniverseSpec`: entity universe
- `TimeSpec`: start/end/calendar/grid/asof settings
- `FeatureRequest`: feature template + params (+ optional slice override)
- `TargetRequest`: target template + params/horizon/name
- `JoinPolicy`: feature join policy (`inner` or `outer`)
- `MissingnessPolicy`: final row policy (`drop_if_any_nan` or `keep`)

## Typical build flow

1. Create `DataContext` with sources, calendar, and store.
2. Define features and target requests.
3. Assemble `DatasetSpec`.
4. Call `build_dataset(ctx, spec, persist=True)`.
5. Consume `DatasetArtifact` (`X`, `y`, `catalog`, metadata).

## Slice overrides

Use `SliceOverride` on a per-feature/per-target basis when a request needs a different lookback, grid, or as-of value than the global spec.

## Output contract

`build_dataset` returns a `DatasetArtifact` with:

- `X`: `pd.DataFrame` indexed by `(ts_utc, entity_id)`
- `y`: `pd.Series` aligned to `X`
- `catalog`: feature catalog dataframe
- `meta` / `aux`: metadata payloads

See the [API reference](../api/dataset-spec.md) for full typed fields.
