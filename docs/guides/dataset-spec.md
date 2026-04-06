# Dataset Spec Guide

`DatasetSpec` is the declarative contract for reproducible dataset builds.

For new research assembly code, treat it as the canonical dataset-building
surface.

## Main components

- `UniverseSpec`: entity universe
- `TimeSpec`: start/end/calendar/grid/asof settings
- `FeatureRequest`: feature template + params (+ optional slice override)
- `FeatureRequestGroup`: composable group of feature requests with inherited
  tags, key prefixes, and slice overrides
- `TargetRequest`: target template + params/horizon/name
- `JoinPolicy`: feature join policy (`inner` or `outer`)
- `MissingnessPolicy`: final row policy (`drop_if_any_nan` or `keep`)

## Typical build flow

1. Create `DataContext` with sources, calendar, and store.
2. Define features and target requests.
3. Assemble `DatasetSpec`.
4. Call `build_dataset(ctx, spec, persist=True)`.
5. Consume `DatasetArtifact` (`X`, `y`, `catalog`, metadata).

## Template composition

`DatasetSpec.features` can contain either flat `FeatureRequest` objects or
nested `FeatureRequestGroup` objects.

Composition rules are explicit:

- group `tags` are inherited by all nested requests
- more specific tags win on key collision:
  outer group -> inner group -> request
- group `slice_override` is inherited per field
- request-level `slice_override` wins per field when both are present
- group `key` prefixes nested request keys with `/`

Example:

```python
from alphaforge.features.dataset_spec import FeatureRequest, FeatureRequestGroup, SliceOverride

features = [
    FeatureRequestGroup(
        key="macro",
        tags={"family": "macro", "recipe": "volatility"},
        slice_override=SliceOverride(lookback=pd.Timedelta(days=30)),
        requests=[
            FeatureRequest(
                template=CarryTemplate(),
                key="carry",
                tags={"series": "carry"},
            ),
            FeatureRequest(
                template=InflationTemplate(),
                key="inflation",
                tags={"series": "cpi"},
            ),
        ],
    )
]
```

The resulting feature catalog records request-level composition metadata such
as:

- `request_key`
- `template_name`
- `template_version`
- merged `tags_json`

## Slice overrides

Use `SliceOverride` on a per-feature/per-target basis when a request needs a different lookback, grid, or as-of value than the global spec.

When a request lives inside a `FeatureRequestGroup`, the group override is
applied first and the request override refines it.

## Join policy

`JoinPolicy` controls how feature families combine before final missingness
handling.

- `inner`: keep only timestamps/entities present across all feature families
- `outer`: union feature-family rows first, then rely on the missingness policy
  to decide what survives

The builder always aligns features onto the explicit evaluation grid defined by
`TimeSpec`, so join policy governs feature-family composition, not whether the
dataset has a deterministic time/entity index.

## Missingness policy

- `drop_if_any_nan`: keep only final rows where every feature column and the
  target are present
- `keep`: preserve the aligned dataset even when some features or target rows
  are missing

## Template behavior expectations

Feature templates are expected to return a `FeatureFrame` whose:

- `X` uses a `MultiIndex` of `(ts_utc, entity_id)`
- `catalog` contains one row per feature id
- output timestamps respect the requested slice semantics, especially `asof`
  for PIT-sensitive templates

The dataset builder preserves request tags, annotates request/template metadata
in the catalog, and keeps leakage detection as a best-effort warning when a
template returns timestamps beyond the requested `asof`.

## Built-in notebook-ready templates

Alphaforge now ships a small built-in template family for common market-price
research work:

- `LagReturnsTemplate`
- `RollingVolatilityTemplate`

These templates use the canonical adapter-backed loading path
(`DataContext.from_adapters(...)` plus `ctx.load(...)`) and are intended to
replace repeated notebook helper cells for lagged returns and trailing
volatility windows.

```python
from alphaforge.features import LagReturnsTemplate, RollingVolatilityTemplate

FeatureRequestGroup(
    key="volatility",
    tags={"recipe": "volatility"},
    requests=[
        FeatureRequest(
            template=LagReturnsTemplate(),
            key="returns",
            params={"dataset": "market.ohlcv", "source": "market", "lags": [1, 5, 10]},
        ),
        FeatureRequest(
            template=RollingVolatilityTemplate(),
            key="trailing_vol",
            params={
                "dataset": "market.ohlcv",
                "source": "market",
                "windows": [5, 10, 21],
                "lag": 1,
                "annualization_factor": 252,
            },
        ),
    ],
)
```

## Output contract

`build_dataset` returns a `DatasetArtifact` with:

- `X`: `pd.DataFrame` indexed by `(ts_utc, entity_id)`
- `y`: `pd.Series` aligned to `X`
- `catalog`: feature catalog dataframe
- `meta` / `aux`: metadata payloads

See the [API reference](../api/dataset-spec.md) for full typed fields.
